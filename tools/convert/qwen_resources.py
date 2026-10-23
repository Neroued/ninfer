"""Qwen text and processor resource binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .resources import TEXT_RESOURCES, load_resources

VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")


def load_qwen_resources(
    model_dir: Path,
    *,
    vocab_size: int,
    vision_config: Mapping[str, int] | None,
    overrides: Mapping[str, str | Path] | None,
):
    roles = {"text": TEXT_RESOURCES}
    if vision_config is not None:
        roles["vision"] = VISION_RESOURCES
    result = load_resources(model_dir, vocab_size=vocab_size, roles=roles, overrides=overrides)
    references, payloads, _, _ = result
    if vision_config is not None:
        for role, object_id in references["vision"].items():
            value = json.loads(payloads[object_id])
            for field, config_field in (
                ("patch_size", "patch_size"),
                ("temporal_patch_size", "temporal_patch_size"),
                ("merge_size", "spatial_merge_size"),
            ):
                if type(value.get(field)) is not int or value[field] != vision_config[config_field]:
                    raise ValueError(f"{role}: {field} differs from Vision config")
    return result
