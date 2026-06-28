#!/usr/bin/env python3
"""Tests for decoded e2e manifest tokenizer path redaction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common  # noqa: E402


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        self.calls.append(skip_special_tokens)
        values = [i for i in ids if not skip_special_tokens or i != 0]
        return "".join(chr(i) for i in values)


class DecodeE2EReportRedactionTests(unittest.TestCase):
    def test_cli_redacts_tokenizer_path_in_manifest_by_default(self) -> None:
        from tools.bench import decode_e2e_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "local-tokenizer"
            tokenizer_path.mkdir()
            report_path = root / "report.json"
            output_dir = root / "decoded"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "qus_e2e_benchmark_report",
                        "status": "ok",
                        "cases": [
                            {
                                "name": "cn_short",
                                "prompt_format": tokenizer_common.PROMPT_FORMAT,
                                "messages_path": "bench/fixtures/prompts/cn_short.messages.json",
                                "messages_sha256": "1" * 64,
                                "rendered_prompt_sha256": "2" * 64,
                                "add_generation_prompt": tokenizer_common.ADD_GENERATION_PROMPT,
                                "add_special_tokens": tokenizer_common.ADD_SPECIAL_TOKENS,
                                "chat_template_kwargs": tokenizer_common.CHAT_TEMPLATE_KWARGS,
                                "stop_token_ids": tokenizer_common.STOP_TOKEN_IDS,
                                "repeats": [
                                    {"repeat_index": 0, "generated_token_ids": [65, 0, 66, 67]},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "decode_e2e_report.py",
                "--report",
                str(report_path),
                "--tokenizer-path",
                str(tokenizer_path),
                "--output-dir",
                str(output_dir),
            ]
            fake = FakeTokenizer()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                decode_e2e_report.common, "load_tokenizer", return_value=fake
            ):
                self.assertEqual(decode_e2e_report.main(), 0)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokenizer"]["tokenizer_path"], "")
            self.assertIn("chat_template_jinja_sha256", manifest["tokenizer"])
            self.assertIn("generation_config_sha256", manifest["tokenizer"])
            self.assertEqual(manifest["prompt_format"], tokenizer_common.PROMPT_FORMAT)
            self.assertEqual(
                manifest["chat_template_kwargs"], tokenizer_common.CHAT_TEMPLATE_KWARGS
            )
            self.assertTrue(manifest["add_generation_prompt"])
            self.assertFalse(manifest["add_special_tokens"])
            self.assertEqual(manifest["stop_token_ids"], tokenizer_common.STOP_TOKEN_IDS)
            self.assertEqual(fake.calls, [False, True])
            artifact = manifest["artifacts"][0]
            raw_path = Path(artifact["raw_text_path"])
            clean_path = Path(artifact["clean_text_path"])
            self.assertTrue(raw_path.exists())
            self.assertTrue(clean_path.exists())
            self.assertEqual(raw_path.name, "repeat_0.raw.txt")
            self.assertEqual(clean_path.name, "repeat_0.clean.txt")
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "A\x00BC")
            self.assertEqual(clean_path.read_text(encoding="utf-8"), "ABC")
            self.assertEqual(artifact["raw_text_chars"], 4)
            self.assertEqual(artifact["clean_text_chars"], 3)
            self.assertEqual(artifact["clean_text_sha256"], tokenizer_common.sha256_text("ABC"))
            self.assertTrue(artifact["clean_text_nonempty_after_strip"])


if __name__ == "__main__":
    unittest.main()
