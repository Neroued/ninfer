from __future__ import annotations

from pathlib import Path

import pytest

from tools.convert.qwen3_6.common.official_resources import (
    OFFICIAL_RESOURCE_SHA256,
    validate_official_resource_hashes,
)
from tools.convert.qwen3_6_27b import convert as convert_27b
from tools.convert.qwen3_6_35b_a3b import convert as convert_35b


MODEL_27B = Path(
    "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16"
)
MODEL_35B = Path(
    "/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16"
)
UNSLOTH_TOKENIZER_SHA256 = (
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
)


@pytest.mark.parametrize(
    ("loader", "model_dir"),
    ((convert_27b.load_resources, MODEL_27B), (convert_35b.load_resources, MODEL_35B)),
)
def test_both_official_sources_pass_the_shared_preflight(loader, model_dir):
    resources = loader(model_dir)

    assert tuple(resource.name for resource in resources) == tuple(
        OFFICIAL_RESOURCE_SHA256
    )


def test_unsloth_tokenizer_hash_is_rejected():
    hashes = dict(OFFICIAL_RESOURCE_SHA256)
    hashes["frontend/tokenizer.json"] = UNSLOTH_TOKENIZER_SHA256

    with pytest.raises(
        ValueError,
        match=(
            "tokenizer.json.*expected "
            + OFFICIAL_RESOURCE_SHA256["frontend/tokenizer.json"]
            + ".*got "
            + UNSLOTH_TOKENIZER_SHA256
        ),
    ):
        validate_official_resource_hashes(hashes)
