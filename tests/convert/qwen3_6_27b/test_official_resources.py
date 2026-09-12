from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.convert.qwen3_6.common.official_resources import (
    OFFICIAL_RESOURCE_SHA256,
    validate_official_resource_hashes,
)
from tools.convert.qwen3_6_27b import convert as convert_27b
from tools.convert.qwen3_6_35b_a3b import convert as convert_35b


UNSLOTH_TOKENIZER_SHA256 = (
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
)


@pytest.mark.parametrize(
    ("loader", "environment"),
    (
        (convert_27b.load_resources, "NINFER_QWEN3_6_27B_MODEL"),
        (convert_35b.load_resources, "NINFER_QWEN3_6_35B_A3B_MODEL"),
    ),
)
def test_both_official_sources_pass_the_shared_preflight(loader, environment):
    model = os.environ.get(environment)
    if not model:
        pytest.skip(f"{environment} is not set")
    resources = loader(Path(model))

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
