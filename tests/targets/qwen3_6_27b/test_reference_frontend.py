from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.reference.qwen3_6.common.frontend import Frontend


MODEL = Path(os.environ.get("NINFER_QWEN3_6_27B_MODEL", ""))
CONFIG_ONLY_TOKENS = {
    "<|audio_start|>": 248070,
    "<|audio_end|>": 248071,
    "<tts_pad>": 248072,
    "<tts_text_bos>": 248073,
    "<tts_text_eod>": 248074,
    "<tts_text_bos_single>": 248075,
    "<|audio_pad|>": 248076,
}


class _OfficialSourceBinding:
    frontend = SimpleNamespace(
        tokenizer_json=MODEL / "tokenizer.json",
        tokenizer_config_json=MODEL / "tokenizer_config.json",
        chat_template_jinja=MODEL / "chat_template.jinja",
        generation_config_json=MODEL / "generation_config.json",
        preprocessor_config_json=MODEL / "preprocessor_config.json",
        video_preprocessor_config_json=MODEL / "video_preprocessor_config.json",
    )

    @staticmethod
    def resource_bytes(resource: Path) -> bytes:
        return resource.read_bytes()


def test_reference_consumes_the_raw_official_resource_pair():
    if not (MODEL / "tokenizer.json").is_file():
        pytest.skip(
            "NINFER_QWEN3_6_27B_MODEL does not point at the Qwen3.6-27B base-hf-bf16 source"
        )

    frontend = Frontend(_OfficialSourceBinding())

    assert len(frontend.tokenizer) == 248077
    assert {
        token: frontend.tokenizer.convert_tokens_to_ids(token)
        for token in CONFIG_ONLY_TOKENS
    } == CONFIG_ONLY_TOKENS
    assert frontend.processor.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
