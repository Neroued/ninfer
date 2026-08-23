"""Build the registered Qwen3.8-27B NVFP4 artifact from its two source roles.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
      --model /path/to/Qwen3.8-27B/base-hf-bf16 \
      --quantized-model /path/to/Qwen3.8-27B/vllm-nvfp4-fp8 \
      --out out/qwen3_8_27b_nvfp4.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import fp8_embedding
from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe


RECIPE_ID = "qwen3_8_27b_nvfp4-v1"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4.ninfer"

_FP8_TARGETS = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
]
_NVFP4_TARGETS = [r"re:.*mlp\.(gate|up|down)_proj$"]


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    official_dir: Path
    quantized_dir: Path
    config_summary: dict[str, object]
    official_source: family_recipe.SourcePreflight
    quantized_source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or not shard
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path}: invalid weight_map entry")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(
            f"{model_dir}: safetensors shard set does not match the index"
        )
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def _validate_float_group(group: Mapping[str, object]) -> None:
    family_conversion.check_members(
        "quantization_config.config_groups.group_0",
        group,
        {"format": "float-quantized", "targets": _FP8_TARGETS},
    )
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, Mapping) or not isinstance(activations, Mapping):
        raise ValueError("FP8 config is missing weight or activation settings")
    common = {
        "num_bits": 8,
        "type": "float",
        "group_size": None,
        "symmetric": True,
        "scale_dtype": None,
    }
    family_conversion.check_members(
        "quantization_config.config_groups.group_0.weights",
        weights,
        {**common, "strategy": "channel", "dynamic": False},
    )
    family_conversion.check_members(
        "quantization_config.config_groups.group_0.input_activations",
        activations,
        {**common, "strategy": "token", "dynamic": True},
    )


def _validate_nvfp4_group(group: Mapping[str, object]) -> None:
    family_conversion.check_members(
        "quantization_config.config_groups.group_1",
        group,
        {"format": "nvfp4-pack-quantized", "targets": _NVFP4_TARGETS},
    )
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, Mapping) or not isinstance(activations, Mapping):
        raise ValueError("NVFP4 config is missing weight or activation settings")
    common = {
        "num_bits": 4,
        "type": "float",
        "strategy": "tensor_group",
        "group_size": 16,
        "symmetric": True,
        "scale_dtype": "torch.float8_e4m3fn",
    }
    family_conversion.check_members(
        "quantization_config.config_groups.group_1.weights",
        weights,
        {**common, "dynamic": False},
    )
    family_conversion.check_members(
        "quantization_config.config_groups.group_1.input_activations",
        activations,
        {**common, "dynamic": "local"},
    )


def _validate_quantized_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    summary = family_config.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("quantized config is missing quantization_config")
    family_conversion.check_members(
        "quantization_config",
        quantization,
        {
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "format": "mixed-precision",
        },
    )
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or tuple(groups) != (
        "group_0",
        "group_1",
    ):
        raise ValueError(
            "quantized config must contain exactly group_0 then group_1"
        )
    float_group = groups["group_0"]
    nvfp4_group = groups["group_1"]
    if not isinstance(float_group, Mapping) or not isinstance(
        nvfp4_group, Mapping
    ):
        raise ValueError("quantized config groups must be objects")
    _validate_float_group(float_group)
    _validate_nvfp4_group(nvfp4_group)
    return summary


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    official_dir: str | Path,
    quantized_dir: str | Path,
) -> ConversionPreflight:
    official = Path(official_dir)
    quantized = Path(quantized_dir)
    _validate_index(official)
    _validate_index(quantized)

    official_config = family_conversion.load_json(official / "config.json")
    if official_config.get("quantization_config") is not None:
        raise ValueError("official source must not declare quantization_config")
    official_summary = family_config.validate_config(official_config)
    quantized_summary = _validate_quantized_config(
        family_conversion.load_json(quantized / "config.json")
    )
    if official_summary != quantized_summary:
        raise ValueError("official and quantized source model configs do not match")
    preflight_inventory()

    with ShardReader(official) as official_reader:
        official_source = recipe.preflight_official_sources(official_reader)
    with ShardReader(quantized) as quantized_reader:
        quantized_source = recipe.preflight_quantized_metadata(quantized_reader)

    resources = base_convert.load_resources(official)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, official)
    return ConversionPreflight(
        official_dir=official,
        quantized_dir=quantized,
        config_summary=official_summary,
        official_source=official_source,
        quantized_source=quantized_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_fp8_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.FP8_WEIGHTS_BY_NAME[spec.name]
    codes, scales = recipe.materialize_fp8_weight(selected, reader)
    return encode_fp8_row_scaled(codes, scales, spec.shape)


def _encode_nvfp4_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _materialize_direct(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    tensor = recipe.materialize_quantized_direct(spec.name, reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    tensor = recipe.materialize_official(spec.name, reader, dict(derived))
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.official_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.official_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "official": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.official_dir.resolve()),
        },
        "quantized": {
            "repository": recipe.QUANTIZED_REPOSITORY,
            "revision": recipe.QUANTIZED_REVISION,
            "model_path": str(preflight.quantized_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "official": {
            "recipes": preflight.official_source.recipe_count,
            "tensors": preflight.official_source.source_tensor_count,
            "shards": preflight.official_source.source_shard_count,
            "dtypes": dict(preflight.official_source.source_dtype_counts),
        },
        "quantized": {
            "recipes": preflight.quantized_source.recipe_count,
            "tensors": preflight.quantized_source.source_tensor_count,
            "shards": preflight.quantized_source.source_shard_count,
            "dtypes": dict(preflight.quantized_source.source_dtype_counts),
            "source_fp8_matrices": len(recipe.FP8_SOURCES),
            "source_nvfp4_matrices": len(recipe.NVFP4_SOURCES),
        },
    }
    report["embedding_encoder"] = fp8_embedding.ENCODER_PROFILE
    return report


def convert(
    official_dir: str | Path,
    quantized_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed dual-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"NVFP4 converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(official_dir, quantized_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.FP8_SOURCES)} FP8 and "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source matrices, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    with ShardReader(preflight.official_dir) as official_reader, ShardReader(
        preflight.quantized_dir
    ) as quantized_reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(
                        official_reader,
                        recipe.OFFICIAL_EMBEDDING_SOURCE.name,
                        spec.shape,
                    )
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = _encode_fp8_weight(spec, quantized_reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = _materialize_direct(spec, quantized_reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    tensor = _materialize_official(
                        spec, official_reader, derived
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(official_dir),
        "quantized_model": str(quantized_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quantized_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
