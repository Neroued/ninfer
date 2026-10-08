"""Check regex membership against independent fullmatch semantics on the shared subset."""

import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(
    os.environ.get("NINFER_REGEX_PROBE", ROOT / "build/tests/ninfer_regex_choice_test")
)


class RegexChoice(unittest.TestCase):
    def test_full_language(self):
        candidates = [
            "".join(chars)
            for n in range(5)
            for chars in itertools.product("ab0 \n", repeat=n)
        ]
        patterns = [
            "",
            "a|ab",
            "|a",
            "a|",
            "a||b",
            "()",
            "(?:)",
            "a(b|a){0,2}",
            "[ab]*",
            "[^a]{1,2}",
            "a?b+",
            "(?:ab)+?",
            "^a|b$",
            "(a|)b?",
            r"\d{1,3}",
            r"[\w\s]{0,2}",
            r"[\D]",
            r"[\W]",
            r"[\S]",
            r"\.",
            "a.*b",
        ]
        request = "".join(
            json.dumps({"regex": pattern, "candidates": candidates}) + "\n"
            for pattern in patterns
        )
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input=request,
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        )
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(reports), len(patterns))
        for pattern, report in zip(patterns, reports):
            with self.subTest(pattern=pattern):
                self.assertNotIn("error", report)
                expected = [
                    re.fullmatch(pattern, text, re.ASCII) is not None
                    for text in candidates
                ]
                self.assertEqual(report["accepted"], expected)


if __name__ == "__main__":
    unittest.main()
