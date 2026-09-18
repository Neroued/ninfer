from __future__ import annotations

import json

import pytest

from tools.convert.resources import load_resources, token_domain


def test_final_resources_override_defaults_without_hash_pinning(tmp_path):
    (tmp_path / "tokenizer.json").write_text("invalid default")
    selected = tmp_path / "custom-tokenizer.json"
    selected.write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
        )
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":2,"temperature":1.0}'
    )
    template = tmp_path / "custom.jinja"
    template.write_text("custom {{ messages }}")
    references, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=4,
        overrides={"tokenizer.json": selected, "chat_template.jinja": template},
    )
    assert count == 3 and special == (2,)
    assert set(references) == {"text"}
    assert payloads[references["text"]["chat_template.jinja"]] == template.read_bytes()
    assert (
        payloads[references["text"]["generation_config.json"]]
        == (tmp_path / "generation_config.json").read_bytes()
    )


def test_special_tokens_merge_both_resources_with_consistent_flags():
    tokenizer = {
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": 1, "content": "<start>", "special": True}],
    }
    config = {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
    assert token_domain(tokenizer, config, 4) == (3, (1, 2))
    config["added_tokens_decoder"]["1"] = {"content": "<start>", "special": False}
    with pytest.raises(ValueError, match="special flag"):
        token_domain(tokenizer, config, 4)
