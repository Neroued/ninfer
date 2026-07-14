"""Convert the registered Qwen3.6-27B checkpoint into one complete artifact.

Canonical invocation::

    python -m tools.convert.qwen3_6_27b_rtx5090.convert \
      --model /path/to/Qwen3.6-27B/base-hf-bf16 \
      --out out/qwen3_6_27b_rtx5090.ninfer
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactObject,
    ArtifactWriter,
    ObjectSpec,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    plan_objects,
)
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device, quantize_and_encode
from tools.convert.common.safetensors import ShardReader

from . import draft_head, inventory, recipe


RECIPE_ID = "qwen3_6_27b_rtx5090-v1"

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "language_model_only": False,
    "mtp_num_hidden_layers": 1,
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 64,
    "full_attention_interval": 4,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 5120,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ObjectPlan:
    specs: tuple[ObjectSpec, ...]
    objects: tuple[ArtifactObject, ...]

    @property
    def payload_span_bytes(self) -> int:
        last = self.objects[-1]
        return last.offset + last.bytes


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(model_dir: Path) -> dict[str, object]:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual.get(name)!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ValueError("checkpoint config mismatch:\n  " + "\n  ".join(mismatches))


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact registered checkpoint dimensions and summarize them."""

    _check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    _check_members("text_config", text, _TEXT_CONFIG)
    expected_layer_types = tuple(
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in range(64)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match the registered 64-layer schedule"
        )
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    _check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    _check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": 64 - len(inventory.FULL_ATTENTION_LAYERS),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
        "mtp_num_hidden_layers": config["mtp_num_hidden_layers"],
        "vision_token_ids": {
            name: config[name]
            for name in (
                "vision_start_token_id",
                "vision_end_token_id",
                "image_token_id",
                "video_token_id",
            )
        },
    }


def preflight_inventory() -> None:
    """Establish the one complete target inventory and recipe pairing."""

    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != (6, 819, 2, 12, 333, 1166, 1172):
        raise ValueError("registered inventory is incomplete")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    root = Path(model_dir)
    resources = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        data = (root / filename).read_bytes()
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        resources.append(ResourcePayload(spec.name, data))
    return tuple(resources)


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    """Compute every payload-relative object offset for the full inventory."""

    preflight_inventory()
    expected_resources = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if tuple(resources) != expected_resources:
        raise ValueError("resource mapping does not match canonical inventory order")
    specs: list[ObjectSpec] = []
    for spec in inventory.OBJECT_SPECS:
        if isinstance(spec, inventory.ResourceSpec):
            specs.append(
                ArtifactResourceSpec(spec.name, spec.encoding, len(resources[spec.name]))
            )
        else:
            specs.append(
                ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout)
            )
    frozen_specs = tuple(specs)
    return ObjectPlan(frozen_specs, plan_objects(frozen_specs))


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    source = recipe.preflight_sources(model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    derived = None
    if spec.name in (
        draft_head.DRAFT_HEAD_OBJECT,
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT,
    ):
        derived = {
            draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
                draft_head.materialize_draft_head_token_ids(draft)
            )
        }
    tensor = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME[spec.name],
        reader,
        derived,
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized tensor according to its registered signature."""

    if spec.format in inventory.DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    return quantize_and_encode(tensor, spec.format, device=device)


def _object_statistics(objects: Sequence[ArtifactObject]) -> dict[str, object]:
    tensors = [obj for obj in objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in objects if isinstance(obj, ResourceObject)]
    object_bytes = sum(obj.bytes for obj in objects)
    payload_span = objects[-1].offset + objects[-1].bytes
    return {
        "count": len(objects),
        "tensors": len(tensors),
        "resources": len(resources),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
        "encodings": dict(
            sorted(Counter(obj.encoding for obj in resources).items())
        ),
        "tensor_bytes": sum(obj.bytes for obj in tensors),
        "resource_bytes": sum(obj.bytes for obj in resources),
        "object_bytes": object_bytes,
        "payload_span_bytes": payload_span,
        "alignment_bytes": payload_span - object_bytes,
    }


def _converter_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _environment(device: torch.device) -> dict[str, object]:
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "resolved_device": str(device),
        "gpu": gpu,
    }


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the external descriptive conversion report."""

    return {
        "model_id": inventory.MODEL_ID,
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {
            "model_path": str(Path(model_dir).resolve()),
            "ranking_path": str(Path(ranking_path).resolve()),
        },
        "arguments": dict(arguments),
        "config_summary": dict(config_summary),
        "source_preflight": {
            "recipes": source_preflight.recipe_count,
            "tensors": source_preflight.source_tensor_count,
            "shards": source_preflight.source_shard_count,
            "dtypes": dict(source_preflight.source_dtype_counts),
        },
        "converter": {
            "revision": _converter_revision() if revision is None else revision,
            "environment": dict(_environment(device) if environment is None else environment),
        },
        "objects": _object_statistics(objects),
        "elapsed_seconds": elapsed_seconds,
        "artifact": {
            "path": str(Path(out_path)),
            "bytes": final_bytes,
        },
    }


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the complete registered conversion and return the report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            inventory.MODEL_ID,
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
