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

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Each case renders through apply_chat_template with its own kwargs. Messages may
# carry reasoning_content and tool_calls; cases may carry tools, enable_thinking,
# and preserve_thinking. This matrix drives the byte-parity gate for the faithful
# render_qwen_chat port (thinking on/off, history strip/keep, reasoning_content
# provided vs derived, tools, tool-call loop, developer role, 1/2/3 systems).
MESSAGES_CASES: list[dict[str, Any]] = [
    {
        "name": "prompt_user_cn_thinking",
        "messages": [{"role": "user", "content": "你好，简单介绍一下你自己。"}],
        "enable_thinking": True,
    },
    {
        "name": "prompt_user_cn_no_thinking",
        "messages": [{"role": "user", "content": "你好，简单介绍一下你自己。"}],
        "enable_thinking": False,
    },
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Describe prefill briefly."},
        ],
        "enable_thinking": False,
    },
    {
        "name": "two_system_merged",
        "messages": [
            {"role": "system", "content": "S1"},
            {"role": "system", "content": "S2"},
            {"role": "user", "content": "hi"},
        ],
        "enable_thinking": False,
    },
    {
        "name": "three_system_drop",
        "messages": [
            {"role": "system", "content": "S1"},
            {"role": "system", "content": "S2"},
            {"role": "system", "content": "S3"},
            {"role": "user", "content": "hi"},
        ],
        "enable_thinking": False,
    },
    {
        "name": "developer_role",
        "messages": [
            {"role": "developer", "content": "D1"},
            {"role": "user", "content": "hi"},
        ],
        "enable_thinking": False,
    },
    {
        "name": "multiturn_strip_thinking",
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>reason1</think>\nA1"},
            {"role": "user", "content": "Q2"},
        ],
        "enable_thinking": True,
    },
    {
        "name": "multiturn_preserve_thinking",
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>reason1</think>\nA1"},
            {"role": "user", "content": "Q2"},
        ],
        "enable_thinking": True,
        "preserve_thinking": True,
    },
    {
        "name": "reasoning_content_provided",
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "myreason"},
            {"role": "user", "content": "Q2"},
        ],
        "enable_thinking": True,
        "preserve_thinking": True,
    },
    {
        "name": "tools_system_no_thinking",
        "messages": [
            {"role": "system", "content": "be direct"},
            {"role": "user", "content": "weather?"},
        ],
        "tools": [WEATHER_TOOL],
        "enable_thinking": False,
    },
    {
        "name": "assistant_toolcall_history",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}],
            },
        ],
        "enable_thinking": True,
    },
    {
        "name": "tool_loop_thinking",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "<think>need tool</think>",
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
            },
            {"role": "tool", "content": '{"temp":20}'},
            {"role": "assistant", "content": "It is 20 degrees in Paris."},
            {"role": "user", "content": "and tomorrow?"},
        ],
        "tools": [WEATHER_TOOL],
        "enable_thinking": True,
    },
]


def to_hf_messages(case_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in case_messages:
        hf: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
        if "reasoning_content" in msg:
            hf["reasoning_content"] = msg["reasoning_content"]
        if "tool_calls" in msg:
            hf["tool_calls"] = [
                {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in msg["tool_calls"]
            ]
        out.append(hf)
    return out


def to_fixture_messages(case_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in case_messages:
        entry: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
        if "reasoning_content" in msg:
            entry["reasoning_content"] = msg["reasoning_content"]
        if "tool_calls" in msg:
            entry["tool_calls"] = [
                {
                    "name": tc["name"],
                    "arguments_json": json.dumps(tc["arguments"], ensure_ascii=False),
                }
                for tc in msg["tool_calls"]
            ]
        out.append(entry)
    return out


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
        hf_messages = to_hf_messages(case["messages"])
        tools = case.get("tools")
        enable_thinking = bool(case.get("enable_thinking", True))
        preserve_thinking = bool(case.get("preserve_thinking", False))
        tool_jsons = [json.dumps(t, ensure_ascii=False) for t in (tools or [])]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
            "return_dict": False,
        }
        if tools is not None:
            kwargs["tools"] = tools
        rendered = tok.apply_chat_template(hf_messages, **kwargs)
        # Guard the tojson key-order assumption the C++ side relies on: the exact
        # serialized tool string must appear verbatim in the rendered prompt.
        for tj in tool_jsons:
            if tj not in rendered:
                raise SystemExit(
                    f"tool json serialization mismatch in case {case['name']!r}: {tj!r}"
                )
        kwargs["tokenize"] = True
        ids = tok.apply_chat_template(hf_messages, **kwargs)
        message_cases.append(
            {
                "name": case["name"],
                "messages": to_fixture_messages(case["messages"]),
                "tool_jsons": tool_jsons,
                "enable_thinking": enable_thinking,
                "preserve_thinking": preserve_thinking,
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
