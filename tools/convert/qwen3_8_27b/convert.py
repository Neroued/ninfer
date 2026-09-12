"""Convert the registered Qwen3.8-27B checkpoint into one complete artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert \
      --model /path/to/Qwen3.8-27B \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
from tools.convert.qwen3_6_27b import draft_head, recipe

from . import dflash2_recipe, inventory
from .dflash2_inventory import DFLASH2_TENSOR_SPECS


RECIPE_ID = "qwen3_8_27b-v2"
BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    ),
    "frontend/tokenizer_config.json": (
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
    ),
    "frontend/chat_template.jinja": (
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    ),
    "frontend/generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    dflash2_model_dir: Path
    base_config_summary: dict[str, object]
    dflash2_config_summary: dict[str, object]
    base_source: recipe.SourcePreflight
    dflash2_source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preflight_inventory() -> None:
    family_recipe.validate_recipe_coverage(
        recipe.RECIPE_SPECS, inventory.BASE_TENSOR_SPECS
    )
    dflash2_recipe.validate_recipe_coverage()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    spec_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official Qwen3.8 profile: "
            f"expected {expected_names!r}, got {spec_names!r}"
        )
    resources = family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    actual_names = tuple(resource.name for resource in resources)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.8 frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for resource in resources:
        actual = hashlib.sha256(resource.data).hexdigest()
        expected = OFFICIAL_RESOURCE_SHA256[resource.name]
        if actual != expected:
            filename = resource.name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.8 resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
    return resources


def preflight_conversion(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
) -> ConversionPreflight:
    model = Path(model_dir)
    dflash2_model = Path(dflash2_model_dir)
    base_config_summary = qwen3_6_convert.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    dflash2_config_summary = dflash2_recipe.validate_config(
        family_conversion.load_json(dflash2_model / "config.json")
    )
    dflash2_recipe.validate_base_compatibility(
        base_config_summary, dflash2_config_summary
    )
    preflight_inventory()
    base_source = recipe.preflight_sources(model)
    dflash2_source = dflash2_recipe.preflight_sources(dflash2_model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        dflash2_model_dir=dflash2_model,
        base_config_summary=base_config_summary,
        dflash2_config_summary=dflash2_config_summary,
        base_source=base_source,
        dflash2_source=dflash2_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    return qwen3_6_convert.materialize_tensor(spec, reader, draft)


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    base_config_summary: Mapping[str, object],
    dflash2_config_summary: Mapping[str, object],
    base_source_preflight: recipe.SourcePreflight,
    dflash2_source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
) -> dict[str, object]:
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary={
            "base": dict(base_config_summary),
            "dflash2": dict(dflash2_config_summary),
        },
        source_preflight=base_source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
    )
    report["source"] = {
        "base": {
            "repository": BASE_REPOSITORY,
            "revision": BASE_REVISION,
            "model_path": str(Path(model_dir).resolve()),
        },
        "dflash2": {
            "repository": dflash2_recipe.REPOSITORY,
            "revision": dflash2_recipe.REVISION,
            "model_path": str(Path(dflash2_model_dir).resolve()),
        },
        "ranking_path": str(Path(ranking_path).resolve()),
    }
    report["source_preflight"] = {
        "base": {
            "recipes": base_source_preflight.recipe_count,
            "tensors": base_source_preflight.source_tensor_count,
            "shards": base_source_preflight.source_shard_count,
            "dtypes": dict(base_source_preflight.source_dtype_counts),
        },
        "dflash2": {
            "recipes": dflash2_source_preflight.recipe_count,
            "tensors": dflash2_source_preflight.source_tensor_count,
            "shards": dflash2_source_preflight.source_shard_count,
            "dtypes": dict(dflash2_source_preflight.source_dtype_counts),
        },
    }
    return report


def convert(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, dflash2_model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.base_source.source_tensor_count} base and "
        f"{preflight.dflash2_source.source_tensor_count} DFlash2 source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(inventory.OBJECT_SPECS)
    index = 0
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer object plan differs from completed preflight")

        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)

        with ShardReader(model) as base_reader:
            for spec in inventory.BASE_TENSOR_SPECS:
                index += 1
                tensor = materialize_tensor(spec, base_reader, preflight.draft)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{total}] {spec.name}",
                    flush=True,
                )

        with ShardReader.from_file(
            preflight.dflash2_model_dir / "model.safetensors"
        ) as dflash2_reader:
            for spec in DFLASH2_TENSOR_SPECS:
                index += 1
                tensor = dflash2_recipe.materialize_tensor(
                    spec.name, dflash2_reader
                )
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "dflash2_model": str(dflash2_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        model_dir=model,
        dflash2_model_dir=preflight.dflash2_model_dir,
        out_path=output,
        arguments=arguments,
        base_config_summary=preflight.base_config_summary,
        dflash2_config_summary=preflight.dflash2_config_summary,
        base_source_preflight=preflight.base_source,
        dflash2_source_preflight=preflight.dflash2_source,
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
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.dflash2_model, args.out, device=args.device)


if __name__ == "__main__":
    main()
