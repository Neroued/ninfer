"""Behavior of the explicitly selected tools/chat_templates files and their C++ rendering."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "build" / "tests" / "ninfer_jinja_test"
SOURCES = {
    version: (ROOT / "tools" / "chat_templates" / f"{version}.jinja").read_text()
    for version in ("qwen3_6", "qwen3_8", "qwen3_8_flash_next")
}


def fail(message):
    raise ValueError(message)


def compile_template(source):
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    # Transformers' chat-template JSON convention, with insertion order preserved.
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, **{"ensure_ascii": False, **kwargs}
    )
    env.globals["raise_exception"] = fail
    return env.from_string(source)


TEMPLATES = {key: compile_template(source) for key, source in SOURCES.items()}


def message(role, content, **fields):
    return {"role": role, "content": content, **fields}


class ChatTemplates(unittest.TestCase):
    def render(self, version, messages, **kwargs):
        return TEMPLATES[version].render(
            messages=messages,
            **{"add_generation_prompt": False, "reasoning_effort": "medium", **kwargs},
        )

    def test_positional_instructions_preserve_history(self):
        history = [message("developer", "policy"), message("user", "question")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                before = self.render(version, history)
                self.assertEqual(
                    before,
                    "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n",
                )
                appended = history + [
                    message("system", "diagnostic"),
                    message("developer", "reminder"),
                ]
                self.assertEqual(
                    self.render(version, appended),
                    before
                    + "<|im_start|>system\ndiagnostic<|im_end|>\n<|im_start|>system\nreminder<|im_end|>\n",
                )

    def test_defaults_and_reasoning_retention(self):
        history = [
            message("user", "first"),
            message("assistant", "answer", reasoning_content="prior reasoning"),
        ]
        next_user = history + [message("user", "second")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                before = self.render(version, history, preserve_thinking=False)
                self.assertIn("prior reasoning", before)
                self.assertTrue(
                    self.render(
                        version,
                        history + [message("system", "diagnostic")],
                        preserve_thinking=False,
                    ).startswith(before)
                )
                self.assertNotIn(
                    "prior reasoning",
                    self.render(version, next_user, preserve_thinking=False),
                )
                self.assertIn(
                    "prior reasoning",
                    self.render(version, next_user, preserve_thinking=True),
                )
        self.assertNotIn("prior reasoning", self.render("qwen3_6", next_user))
        self.assertIn("prior reasoning", self.render("qwen3_8", next_user))
        default = TEMPLATES["qwen3_8"].render(
            messages=[message("user", "hello")], add_generation_prompt=True
        )
        self.assertIn("Reasoning effort is set to xhigh.", default)
        self.assertTrue(default.endswith("<|im_start|>assistant\n<think>\n"))

    def test_tools_and_instruction_preamble(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "inspect", "parameters": {"type": "object"}},
            }
        ]
        call = {
            "type": "function",
            "function": {
                "name": "inspect",
                "arguments": {"city": "北京", "enabled": True},
            },
        }
        history = [
            message("system", "policy"),
            message("user", "inspect"),
            message("assistant", "", tool_calls=[call]),
            message("tool", "one"),
            message("tool", "two"),
            message("developer", "diagnostic"),
        ]
        for version in TEMPLATES:
            with self.subTest(template=version):
                text = self.render(version, history, tools=tools)
                self.assertEqual(text.count("# Tools"), 1)
                self.assertIn(
                    "<parameter=city>\n北京\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>",
                    text,
                )
                self.assertIn(
                    "<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>",
                    text,
                )
                self.assertTrue(
                    text.endswith("<|im_start|>system\ndiagnostic<|im_end|>\n")
                )

    def test_tool_result_without_original_user(self):
        results = [message("tool", "one"), message("tool", "two")]
        before = message("assistant", "first", reasoning_content="before reasoning")
        after = message("assistant", "second", reasoning_content="after reasoning")
        for version in TEMPLATES:
            with self.subTest(template=version):
                self.assertEqual(
                    self.render(version, results),
                    "<|im_start|>user\n<tool_response>\none\n</tool_response>"
                    "\n<tool_response>\ntwo\n</tool_response><|im_end|>\n",
                )
                text = self.render(
                    version,
                    [results[0], before, results[1], after],
                    preserve_thinking=False,
                )
                self.assertNotIn("before reasoning", text)
                self.assertIn("after reasoning", text)
                self.assertIn(
                    "before reasoning",
                    self.render(
                        version,
                        [message("user", "question"), before, results[0]],
                        preserve_thinking=False,
                    ),
                )

    def test_final_assistant_continuation(self):
        history = [message("user", "question"), message("assistant", "answer prefix")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                text = self.render(
                    version, history, continue_final_message=True, enable_thinking=False
                )
                self.assertEqual(
                    text,
                    "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer prefix",
                )
                literal = [
                    message("user", "question"),
                    message("assistant", "prefix </think> text"),
                ]
                self.assertTrue(
                    self.render(
                        version,
                        literal,
                        continue_final_message=True,
                        enable_thinking=False,
                    ).endswith("<|im_start|>assistant\nprefix </think> text")
                )

    def test_cpp_matches_independent_renderer(self):
        contexts = [
            dict(messages=[message("user", "你好🌏")], add_generation_prompt=True),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            dict(
                messages=[
                    message("system", " policy "),
                    message("user", "hi"),
                    message("developer", "late"),
                ],
                add_generation_prompt=True,
                reasoning_effort="low",
            ),
            dict(
                messages=[
                    message(
                        "user",
                        [
                            {"type": "text", "text": "look"},
                            {"type": "image"},
                            {"type": "video"},
                        ],
                    )
                ],
                add_generation_prompt=True,
                add_vision_id=True,
            ),
            dict(
                messages=[
                    message("user", "first"),
                    message("assistant", "answer", reasoning_content="reason"),
                    message("user", "next"),
                ],
                add_generation_prompt=True,
                preserve_thinking=False,
            ),
            dict(
                messages=[message("user", "first"), message("assistant", "prefix")],
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            ),
            dict(
                messages=[message("tool", "one"), message("tool", "two")],
                add_generation_prompt=True,
            ),
            dict(
                messages=[
                    message("tool", "one"),
                    message("assistant", "first", reasoning_content="before reasoning"),
                    message("tool", "two"),
                    message("assistant", "second", reasoning_content="after reasoning"),
                ],
                add_generation_prompt=True,
                preserve_thinking=False,
            ),
            dict(messages=[message("system", "no user")], add_generation_prompt=True),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                reasoning_effort="high",
            ),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                reasoning_effort="max",
            ),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                reasoning_effort="minimal",
            ),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                reasoning_effort="none",
            ),
            dict(
                messages=[message("user", "hi")],
                add_generation_prompt=True,
                reasoning_effort="none",
                enable_thinking=True,
            ),
            dict(
                messages=[
                    message("user", "code"),
                    message("assistant", "  def f():\n    "),
                ],
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            ),
        ]
        cases = [(name, context) for name in SOURCES for context in contexts]
        result = subprocess.run(
            [str(RENDERER), "--render"],
            text=True,
            capture_output=True,
            check=True,
            input="".join(
                json.dumps(
                    {"source": SOURCES[name], "context": context}, ensure_ascii=False
                )
                + "\n"
                for name, context in cases
            ),
            timeout=30,
        )
        actual = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(actual), len(cases))
        for (name, context), got in zip(cases, actual):
            with self.subTest(template=name, context=context):
                try:
                    expected = TEMPLATES[name].render(**context)
                except ValueError:
                    self.assertFalse(got["ok"])
                else:
                    self.assertTrue(got["ok"], got.get("error"))
                    self.assertEqual(got["text"], expected)

    def test_flash_next_effort_aliases_and_defaults(self):
        version = "qwen3_8_flash_next"
        history = [message("user", "question")]
        for alias, native in (("high", "xhigh"), ("max", "xhigh"), ("minimal", "low")):
            self.assertEqual(
                self.render(
                    version, history, reasoning_effort=alias, add_generation_prompt=True
                ),
                self.render(
                    version,
                    history,
                    reasoning_effort=native,
                    add_generation_prompt=True,
                ),
            )
        self.assertEqual(
            self.render(
                version, history, reasoning_effort="none", add_generation_prompt=True
            ),
            self.render(
                version, history, enable_thinking=False, add_generation_prompt=True
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.render(version, history, reasoning_effort="none", enable_thinking=True)
        default = TEMPLATES[version].render(
            messages=history, add_generation_prompt=True
        )
        self.assertIn("Reasoning effort is set to xhigh.", default)
        self.assertEqual(
            default,
            TEMPLATES["qwen3_8"].render(messages=history, add_generation_prompt=True),
        )

    def test_flash_next_continuation_preserves_whitespace(self):
        content = "  def f():\n    "
        text = self.render(
            "qwen3_8_flash_next",
            [message("user", "code"), message("assistant", content)],
            continue_final_message=True,
            enable_thinking=False,
        )
        self.assertEqual(
            text, "<|im_start|>user\ncode<|im_end|>\n<|im_start|>assistant\n" + content
        )

    def test_flash_next_generated_history_preserves_prefix(self):
        version = "qwen3_8_flash_next"
        user = message("user", "question")
        for thinking in (False, True):
            prologue = self.render(
                version, [user], enable_thinking=thinking, add_generation_prompt=True
            )
            reasoning = "reason" if thinking else ""
            generated = (
                "reason\n</think>\n\n" if thinking else ""
            ) + "answer<|im_end|>\n"
            history = [
                user,
                message("assistant", "answer", reasoning_content=reasoning),
            ]
            self.assertEqual(
                self.render(version, history, enable_thinking=thinking),
                prologue + generated,
            )
            self.assertTrue(
                self.render(
                    version,
                    history + [message("user", "next")],
                    enable_thinking=thinking,
                ).startswith(prologue + generated)
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        RENDERER = Path(sys.argv.pop(1))
    unittest.main()
