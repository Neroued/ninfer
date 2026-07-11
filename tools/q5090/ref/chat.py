"""Single-turn Qwen3.6 chat prompt rendering for reference inference."""

from __future__ import annotations


def render_prompt(text: str, *, thinking: bool) -> str:
    text = text.strip()
    if not text:
        raise ValueError("prompt text must not be empty")
    rendered = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    if not thinking:
        rendered += "\n</think>\n\n"
    return rendered
