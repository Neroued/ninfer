#!/usr/bin/env python3
"""Unit tests for M2.8 bench tokenizer/decode helpers."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common as common  # noqa: E402


class FakeTokenizer:
    def __init__(self) -> None:
        self.encoded: dict[str, list[int]] = {}
        self.chat_template_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> str | list[int]:
        if not add_generation_prompt:
            raise AssertionError("fixtures must add a generation prompt")
        if enable_thinking:
            raise AssertionError("fixtures must disable thinking")
        if return_dict:
            raise AssertionError("fixtures must request raw ids/text")
        self.chat_template_calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "return_dict": return_dict,
            }
        )
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        rendered += "<assistant>"
        if not tokenize:
            return rendered
        if rendered in self.encoded:
            return self.encoded[rendered]
        return [ord(ch) for ch in rendered]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        values = [i for i in ids if not skip_special_tokens or i != 0]
        return "".join(chr(i) for i in values)


class TokenizerCommonTests(unittest.TestCase):
    def test_ids_roundtrip_and_rejects_metadata(self) -> None:
        ids = [1, 2, 3, 4096, 151643]
        text = common.format_ids(ids)
        self.assertEqual(common.parse_ids_text(text), ids)
        self.assertEqual(common.parse_ids_text("1 2\n3\t4\n"), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            common.parse_ids_text("")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 -2 3")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 2 # 3")
        with self.assertRaises(ValueError):
            common.parse_ids_text("1 two 3")

    def test_tokenizer_metadata_hashes_chat_template_jinja_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = root / "tokenizer.json"
            config = root / "tokenizer_config.json"
            special = root / "special_tokens_map.json"
            generation = root / "generation_config.json"
            chat_template = root / "chat_template.jinja"
            tokenizer.write_text("tokenizer\n", encoding="utf-8")
            config.write_text(
                json.dumps({"chat_template": "embedded template"}) + "\n",
                encoding="utf-8",
            )
            special.write_text("special\n", encoding="utf-8")
            generation.write_text("generation\n", encoding="utf-8")
            chat_template.write_text("file template\n", encoding="utf-8")

            metadata = common.tokenizer_metadata(root)
            self.assertEqual(metadata["tokenizer_source"], "local_hf")
            self.assertEqual(metadata["tokenizer_model_id"], "Qwen/Qwen3.6-27B")
            self.assertEqual(metadata["tokenizer_path"], str(root))
            self.assertEqual(metadata["tokenizer_json_sha256"], common.sha256_file(tokenizer))
            self.assertEqual(metadata["tokenizer_config_sha256"], common.sha256_file(config))
            self.assertEqual(metadata["special_tokens_map_sha256"], common.sha256_file(special))
            self.assertEqual(
                metadata["chat_template_jinja_sha256"],
                common.sha256_file(chat_template),
            )
            self.assertEqual(metadata["generation_config_sha256"], common.sha256_file(generation))

    def test_tokenizer_metadata_leaves_chat_template_hash_empty_when_file_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "embedded template"}) + "\n",
                encoding="utf-8",
            )
            metadata = common.tokenizer_metadata(root)
            self.assertEqual(metadata["chat_template_jinja_sha256"], "")

    def test_read_messages_accepts_chat_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.messages.json"
            messages = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Explain prefill."},
                {"role": "assistant", "content": "Prefill processes prompt tokens."},
                {"role": "user", "content": "Now summarize."},
            ]
            path.write_text(json.dumps(messages), encoding="utf-8")
            self.assertEqual(common.read_messages(path), messages)

    def test_read_messages_rejects_non_chat_content(self) -> None:
        invalid_values = [
            {},
            [],
            [{"role": ["user"], "content": "x"}],
            [{"role": "tool", "content": "x"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            [{"role": "user", "content": {"type": "text", "text": "x"}}],
            [{"role": "user", "content": "x", "tool_calls": []}],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, value in enumerate(invalid_values):
                path = root / f"bad_{index}.messages.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        common.read_messages(path)

    def test_resolve_tokenizer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(common.resolve_tokenizer_path(str(root)), root)
            old = os.environ.get("QUS_TOKENIZER_PATH")
            try:
                os.environ["QUS_TOKENIZER_PATH"] = str(root)
                self.assertEqual(common.resolve_tokenizer_path(None), root)
            finally:
                if old is None:
                    os.environ.pop("QUS_TOKENIZER_PATH", None)
                else:
                    os.environ["QUS_TOKENIZER_PATH"] = old
            with self.assertRaises(RuntimeError):
                old = os.environ.pop("QUS_TOKENIZER_PATH", None)
                try:
                    common.resolve_tokenizer_path(None)
                finally:
                    if old is not None:
                        os.environ["QUS_TOKENIZER_PATH"] = old


class FixtureManifestTests(unittest.TestCase):
    def test_committed_prompt_messages_use_common_user_questions(self) -> None:
        fixture_dir = common.repo_root() / "bench/fixtures/prompts"
        expected_snippets = {
            "cn_short": "为什么每天适量喝水很重要",
            "en_short": "backing up important files",
            "code_short": "def average(nums):",
            "math_short": "一个杯子 12 元",
            "long_2k": "周末旅行规划资料",
        }
        forbidden = ("M2.8", "benchmark", "prefill", "decode", "q5090")
        for name, snippet in expected_snippets.items():
            with self.subTest(name=name):
                messages = common.read_messages(fixture_dir / f"{name}.messages.json")
                self.assertEqual([message["role"] for message in messages], ["user"])
                text = messages[0]["content"]
                self.assertIn(snippet, text)
                if name != "long_2k":
                    self.assertNotIn("基准测试", text)
                for term in forbidden:
                    self.assertNotIn(term, text)

    def test_build_manifest_records_cases_in_required_order(self) -> None:
        from tools.bench import tokenize_prompts

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "prompts"
            fixture_dir.mkdir()
            fake = FakeTokenizer()
            expected_messages: dict[str, list[dict[str, str]]] = {}
            expected_ids: dict[str, list[int]] = {}
            for index, name in enumerate(common.REQUIRED_CASES):
                messages = [{"role": "user", "content": f"{name} fixture text"}]
                rendered = fake.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                expected_messages[name] = messages
                expected_ids[name] = (
                    list(range(2050)) if name == "long_2k" else [index + 1, index + 2, index + 3]
                )
                fake.encoded[str(rendered)] = expected_ids[name]
                (fixture_dir / f"{name}.messages.json").write_text(
                    json.dumps(messages),
                    encoding="utf-8",
                )
            fake.chat_template_calls.clear()

            metadata = {
                "tokenizer_source": "local_hf",
                "tokenizer_model_id": common.TOKENIZER_MODEL_ID,
                "tokenizer_path": "",
                "tokenizer_json_sha256": "tok",
                "tokenizer_config_sha256": "cfg",
                "special_tokens_map_sha256": "special",
                "chat_template_jinja_sha256": "chat",
                "generation_config_sha256": "gen",
            }
            manifest = tokenize_prompts.write_fixtures(
                fixture_dir=fixture_dir,
                tokenizer=fake,
                tokenizer_metadata=metadata,
                check=False,
            )
            self.assertEqual(manifest["fixture_set"], common.FIXTURE_SET)
            self.assertEqual(manifest["tokenizer"], metadata)
            self.assertEqual(
                manifest["generation"],
                {
                    "stop_token_ids": common.STOP_TOKEN_IDS,
                    "stop_token_names": common.STOP_TOKEN_NAMES,
                    "sampling_policy": "Fixture ids are prompt-only chat-template inputs; decode sampling is configured by benchmark callers.",
                },
            )
            self.assertEqual([case["name"] for case in manifest["cases"]], list(common.REQUIRED_CASES))
            for case in manifest["cases"]:
                name = case["name"]
                ids = common.read_ids(fixture_dir / f"{name}.ids")
                messages_path = fixture_dir / f"{name}.messages.json"
                rendered = str(
                    fake.apply_chat_template(
                        expected_messages[name],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_dict=False,
                    )
                )
                self.assertEqual(case["messages"], f"{name}{common.MESSAGE_FILE_SUFFIX}")
                self.assertEqual(case["ids"], f"{name}.ids")
                self.assertEqual(case["prompt_tokens"], len(ids))
                self.assertEqual(case["messages_sha256"], common.sha256_file(messages_path))
                self.assertEqual(case["rendered_prompt_sha256"], common.sha256_text(rendered))
                self.assertEqual(case["ids_sha256"], common.sha256_file(fixture_dir / f"{name}.ids"))
                self.assertEqual(case["prompt_format"], common.PROMPT_FORMAT)
                self.assertIs(case["add_generation_prompt"], common.ADD_GENERATION_PROMPT)
                self.assertIs(case["add_special_tokens"], common.ADD_SPECIAL_TOKENS)
                self.assertEqual(case["chat_template_kwargs"], common.CHAT_TEMPLATE_KWARGS)
            tokenizing_calls = [call for call in fake.chat_template_calls if call["tokenize"] is True]
            rendering_calls = [call for call in fake.chat_template_calls if call["tokenize"] is False]
            self.assertGreaterEqual(len(tokenizing_calls), len(common.REQUIRED_CASES))
            self.assertGreaterEqual(len(rendering_calls), len(common.REQUIRED_CASES))

            checked = tokenize_prompts.write_fixtures(
                fixture_dir=fixture_dir,
                tokenizer=fake,
                tokenizer_metadata=metadata,
                check=True,
            )
            self.assertEqual(checked, manifest)

    def test_check_mode_rejects_stale_ids(self) -> None:
        from tools.bench import tokenize_prompts

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "prompts"
            fixture_dir.mkdir()
            fake = FakeTokenizer()
            for index, name in enumerate(common.REQUIRED_CASES):
                messages = [{"role": "user", "content": f"{name} fixture text"}]
                rendered = fake.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                fake.encoded[str(rendered)] = (
                    list(range(2050)) if name == "long_2k" else [index + 10, index + 11]
                )
                (fixture_dir / f"{name}.messages.json").write_text(
                    json.dumps(messages),
                    encoding="utf-8",
                )
            metadata = common.tokenizer_metadata(Path(tmp), redact_path=True)
            tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=False)
            (fixture_dir / "cn_short.ids").write_text("1 2 3\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=True)

    def test_check_mode_rejects_stale_messages(self) -> None:
        from tools.bench import tokenize_prompts

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "prompts"
            fixture_dir.mkdir()
            fake = FakeTokenizer()
            for index, name in enumerate(common.REQUIRED_CASES):
                messages = [{"role": "user", "content": f"{name} fixture text"}]
                rendered = fake.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                fake.encoded[str(rendered)] = (
                    list(range(2050)) if name == "long_2k" else [index + 20, index + 21]
                )
                (fixture_dir / f"{name}.messages.json").write_text(
                    json.dumps(messages),
                    encoding="utf-8",
                )
            metadata = common.tokenizer_metadata(Path(tmp), redact_path=True)
            tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=False)
            (fixture_dir / "cn_short.messages.json").write_text(
                json.dumps([{"role": "user", "content": "changed text"}]),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                tokenize_prompts.write_fixtures(fixture_dir, fake, metadata, check=True)


class LongPromptGeneratorTests(unittest.TestCase):
    def test_make_long_prompt_writes_user_messages_json(self) -> None:
        from tools.bench import make_long_prompt

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "long_2k.messages.json"
            rc = make_long_prompt.main(["--out", str(out), "--repeats", "2"])
            self.assertEqual(rc, 0)
            messages = common.read_messages(out)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            self.assertIn("周末旅行规划资料", messages[0]["content"])
            self.assertIn(
                "请根据上面的资料，给一家三口安排一个周六下午到晚上的简短行程。",
                messages[0]["content"],
            )
            self.assertNotIn("M2.8", messages[0]["content"])
            self.assertNotIn("benchmark", messages[0]["content"].lower())


class DecodeReportTests(unittest.TestCase):
    def _report(self, repeats: list[object]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "qus_e2e_benchmark_report",
            "status": "ok",
            "cases": [
                {
                    "name": "cn_short",
                    "prompt_format": common.PROMPT_FORMAT,
                    "messages_path": "bench/fixtures/prompts/cn_short.messages.json",
                    "messages_sha256": "1" * 64,
                    "rendered_prompt_sha256": "2" * 64,
                    "add_generation_prompt": common.ADD_GENERATION_PROMPT,
                    "add_special_tokens": common.ADD_SPECIAL_TOKENS,
                    "chat_template_kwargs": common.CHAT_TEMPLATE_KWARGS,
                    "stop_token_ids": common.STOP_TOKEN_IDS,
                    "repeats": repeats,
                }
            ],
        }

    def test_decode_report_writes_sidecar_manifest_and_text(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            report = self._report(
                [
                    {"repeat_index": 0, "generated_token_ids": [65, 0, 66, 67]},
                    {"repeat_index": 1, "generated_token_ids": [68, 69]},
                ]
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            metadata = {
                "tokenizer_source": "local_hf",
                "tokenizer_model_id": common.TOKENIZER_MODEL_ID,
                "tokenizer_path": "/tmp/tokenizer",
                "tokenizer_json_sha256": "tok",
                "tokenizer_config_sha256": "cfg",
                "special_tokens_map_sha256": "special",
                "chat_template_jinja_sha256": "chat",
                "generation_config_sha256": "gen",
            }
            manifest = decode_e2e_report.decode_report(
                report_path=report_path,
                tokenizer=FakeTokenizer(),
                tokenizer_metadata=metadata,
                output_dir=None,
            )
            self.assertEqual(manifest["artifact_type"], "qus_decoded_text_artifacts")
            self.assertEqual(manifest["readability_gate"], "human_smoke_only")
            self.assertEqual(manifest["source_report"], str(report_path))
            self.assertEqual(len(manifest["artifacts"]), 2)
            manifest_path = root / "report.decoded" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["prompt_format"], common.PROMPT_FORMAT)
            self.assertEqual(manifest["chat_template_kwargs"], common.CHAT_TEMPLATE_KWARGS)
            first_raw = Path(manifest["artifacts"][0]["raw_text_path"])
            first_clean = Path(manifest["artifacts"][0]["clean_text_path"])
            self.assertEqual(first_raw.name, "repeat_0.raw.txt")
            self.assertEqual(first_clean.name, "repeat_0.clean.txt")
            self.assertEqual(first_raw.read_text(encoding="utf-8"), "A\x00BC")
            self.assertEqual(first_clean.read_text(encoding="utf-8"), "ABC")
            self.assertEqual(manifest["artifacts"][0]["raw_text_chars"], 4)
            self.assertEqual(manifest["artifacts"][0]["clean_text_chars"], 3)
            self.assertEqual(
                manifest["artifacts"][0]["clean_text_sha256"],
                common.sha256_text("ABC"),
            )
            self.assertTrue(manifest["artifacts"][0]["clean_text_nonempty_after_strip"])

    def test_decode_report_rejects_bad_repeat_records_with_runtime_error(self) -> None:
        from tools.bench import decode_e2e_report

        bad_reports = {
            "non_dict_repeat": self._report(["not-a-repeat"]),
            "bad_repeat_index": self._report(
                [{"repeat_index": "not-an-int", "generated_token_ids": [65]}]
            ),
            "bool_repeat_index": self._report(
                [{"repeat_index": True, "generated_token_ids": [65]}]
            ),
            "bool_generated_token_id": self._report(
                [{"repeat_index": 0, "generated_token_ids": [65, True]}]
            ),
            "empty_generated_token_ids": self._report(
                [{"repeat_index": 0, "generated_token_ids": []}]
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, report in bad_reports.items():
                with self.subTest(name=name):
                    report_path = root / f"{name}.json"
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "repeat"):
                        decode_e2e_report.decode_report(
                            report_path=report_path,
                            tokenizer=FakeTokenizer(),
                            tokenizer_metadata=common.tokenizer_metadata(
                                Path(tmp),
                                redact_path=True,
                            ),
                            output_dir=None,
                        )

    def test_decode_report_rejects_bad_case_chat_identity(self) -> None:
        from tools.bench import decode_e2e_report

        def stale_eos_token_id() -> dict[str, object]:
            report = self._report([{"repeat_index": 0, "generated_token_ids": [65]}])
            report["cases"][0]["eos_token_id"] = -1
            return report

        def bool_stop_token_ids() -> dict[str, object]:
            report = self._report([{"repeat_index": 0, "generated_token_ids": [65]}])
            report["cases"][0]["stop_token_ids"] = [True]
            return report

        def bad_prompt_format() -> dict[str, object]:
            report = self._report([{"repeat_index": 0, "generated_token_ids": [65]}])
            report["cases"][0]["prompt_format"] = "raw-text"
            return report

        def bad_chat_template_kwargs() -> dict[str, object]:
            report = self._report([{"repeat_index": 0, "generated_token_ids": [65]}])
            report["cases"][0]["chat_template_kwargs"] = {"enable_thinking": 0}
            return report

        bad_reports = {
            "stale_eos_token_id": stale_eos_token_id(),
            "bool_stop_token_ids": bool_stop_token_ids(),
            "bad_prompt_format": bad_prompt_format(),
            "bad_chat_template_kwargs": bad_chat_template_kwargs(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, report in bad_reports.items():
                with self.subTest(name=name):
                    report_path = root / f"{name}.json"
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "case"):
                        decode_e2e_report.decode_report(
                            report_path=report_path,
                            tokenizer=FakeTokenizer(),
                            tokenizer_metadata=common.tokenizer_metadata(
                                Path(tmp),
                                redact_path=True,
                            ),
                            output_dir=None,
                        )

    def test_decode_report_rejects_non_ok_report(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "qus_e2e_benchmark_report",
                        "status": "error",
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                decode_e2e_report.decode_report(
                    report_path=report_path,
                    tokenizer=FakeTokenizer(),
                    tokenizer_metadata=common.tokenizer_metadata(Path(tmp), redact_path=True),
                    output_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
