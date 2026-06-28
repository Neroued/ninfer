#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES: list[dict[str, Any]] = [
    {"name": "ascii", "text": "Hello, world!"},
    {"name": "chinese", "text": "你好，世界！"},
    {"name": "leading_space", "text": " leading space"},
    {"name": "trailing_space", "text": "trailing space "},
    {"name": "multi_space", "text": "alpha  beta   gamma"},
    {"name": "newline", "text": "line1\nline2"},
    {"name": "emoji", "text": "emoji 😀 test"},
    {"name": "combining_nfc", "text": "Cafe\u0301"},
    {"name": "code", "text": "def f(xs):\n    return xs or []\n"},
    {"name": "chat_markers", "text": "<|im_start|>user\nhi<|im_end|>\n"},
    {"name": "thinking_markers", "text": "<think>\n\n</think>\n\n"},
]

MESSAGES_CASES: list[dict[str, Any]] = [
    {
        "name": "prompt_user_cn",
        "messages": [{"role": "user", "content": "你好，简单介绍一下你自己。"}],
    },
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Describe prefill briefly."},
        ],
    },
    {
        "name": "assistant_history",
        "messages": [
            {"role": "user", "content": "Say one word."},
            {"role": "assistant", "content": "Ready."},
            {"role": "user", "content": "Now say two words."},
        ],
    },
]


def load_tokenizer(tokenizer_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer_path)
    text_cases = []
    for case in CASES:
        ids = tok.encode(case["text"], add_special_tokens=False)
        text_cases.append(
            {
                "name": case["name"],
                "text": case["text"],
                "ids": list(ids),
                "raw_decoded": tok.decode(ids, skip_special_tokens=False),
                "clean_decoded": tok.decode(ids, skip_special_tokens=True),
            }
        )

    message_cases = []
    for case in MESSAGES_CASES:
        rendered = tok.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        ids = tok.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        message_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "rendered": rendered,
                "ids": list(ids),
                "raw_decoded": tok.decode(ids, skip_special_tokens=False),
                "clean_decoded": tok.decode(ids, skip_special_tokens=True),
            }
        )

    value = {
        "tokenizer_model_id": "Qwen/Qwen3.6-27B",
        "text_cases": text_cases,
        "message_cases": message_cases,
        "default_stop_token_ids": [248046, 248044],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
