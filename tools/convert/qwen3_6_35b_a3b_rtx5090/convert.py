"""Convert Qwen3.6-35B-A3B BF16 weights into its exact NInfer artifact.

Canonical invocation::

    python -m tools.convert.qwen3_6_35b_a3b_rtx5090.convert \
      --model /home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16 \
      --out out/qwen3_6_35b_a3b_rtx5090.ninfer

The target deliberately reuses the measured 27B ranking because both checkpoints
have the same semantic token-id vocabulary.  Draft rows are always gathered from
the 35B output head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import draft_head, inventory, recipe


RECIPE_ID = "qwen3_6_35b_a3b_rtx5090-v1"
ENCODER_PROFILE = "MAXABS_F16_RECIP_RNE_V1"
GGUF_EVIDENCE_PATH = Path(
    "/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/"
    "gguf-ud-q4_k_m/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 40,
    "full_attention_interval": 4,
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attn_output_gate": True,
    "hidden_act": "silu",
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "max_position_embeddings": 262144,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
    "partial_rotary_factor": 0.25,
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 2048,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}

EXPECTED_TENSOR_BYTES = 22_360_191_904
EXPECTED_DEVICE_ARENA_BYTES = 22_360_207_360
EXPECTED_COMPONENT_BYTES = {
    "main_text": 21_038_461_952,
    "draft_head": 143_130_624,
    "mtp": 897_934_336,
    "vision": 280_664_992,
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


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


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every checkpoint fact that fixes storage or execution shape."""

    family_conversion.check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)

    expected_layer_types = tuple(
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in range(40)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match the target 40-layer schedule"
        )

    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": len(inventory.GDN_LAYERS),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
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
    """Prove the target-private inventory before any payload is written."""

    counts = (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    )
    if counts != (6, 533, 2, 15, 333, 883, 889):
        raise ValueError(f"target inventory is incomplete: {counts}")
    if inventory.FORMAT_COUNTS != {
        inventory.BF16: 461,
        inventory.FP32: 60,
        inventory.I32: 1,
        inventory.Q4: 95,
        inventory.Q5: 91,
        inventory.Q6: 5,
        inventory.W8: 170,
    }:
        raise ValueError(f"target format counts drifted: {inventory.FORMAT_COUNTS}")
    if inventory.LAYOUT_COUNTS != {
        inventory.CONTIGUOUS_LAYOUT: 522,
        inventory.ROW_SPLIT_LAYOUT: 361,
    }:
        raise ValueError(f"target layout counts drifted: {inventory.LAYOUT_COUNTS}")
    if family_conversion.tensor_payload_bytes(inventory.TENSOR_SPECS) != EXPECTED_TENSOR_BYTES:
        raise ValueError("target tensor payload byte total drifted")
    if (
        family_conversion.device_arena_bytes(inventory.TENSOR_SPECS)
        != EXPECTED_DEVICE_ARENA_BYTES
    ):
        raise ValueError("target device-arena byte total drifted")
    component_bytes = {
        "main_text": family_conversion.tensor_payload_bytes(
            inventory.TEXT_CORE_TENSOR_SPECS
        ),
        "draft_head": family_conversion.tensor_payload_bytes(
            inventory.DRAFT_HEAD_TENSOR_SPECS
        ),
        "mtp": family_conversion.tensor_payload_bytes(inventory.MTP_TENSOR_SPECS),
        "vision": family_conversion.tensor_payload_bytes(
            inventory.VISION_TENSOR_SPECS
        ),
    }
    if component_bytes != EXPECTED_COMPONENT_BYTES:
        raise ValueError(f"target component byte totals drifted: {component_bytes}")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    """Complete config, source, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config_summary = validate_config(
        family_conversion.load_json(model / "config.json")
    )
    preflight_inventory()
    source = recipe.preflight_sources(model)
    if (
        source.recipe_count,
        source.source_tensor_count,
        source.source_shard_count,
        source.source_dtype_counts,
    ) != (883, 1045, 26, {"BF16": 1045}):
        raise ValueError(f"checkpoint source inventory drifted: {source}")

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
    return family_conversion.encode_tensor_payload(tensor, spec, device)


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
    report = family_conversion.build_conversion_report(
        model_id=inventory.MODEL_ID,
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )
    report["draft_head"] = {
        "rows": draft_head.DRAFT_HEAD_N,
        "tokenizer_vocab_size": draft_head.TOKENIZER_VOCAB_SIZE,
        "ranking_source_target": draft_head.RANKING_SOURCE_TARGET,
        "shared_semantic_vocabulary": True,
    }
    report["source"]["gguf_evidence_path"] = str(GGUF_EVIDENCE_PATH)
    report["quantization"] = {
        "encoder_profile": ENCODER_PROFILE,
        "component_tensor_bytes": {
            **EXPECTED_COMPONENT_BYTES,
            "total": EXPECTED_TENSOR_BYTES,
            "device_arena": EXPECTED_DEVICE_ARENA_BYTES,
        },
    }
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the complete target conversion and return its report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
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
