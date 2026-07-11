import os
import tempfile
import unittest
from pathlib import Path

from qus_eval.config import ConfigError, load_config
from qus_eval.secrets import resolve_target

BASE = """
schema_version: 1
targets:
  api:
    protocol: openai_chat
    base_url: http://127.0.0.1:8000/v1
    model: test-model
    max_concurrency: 2
suites:
  test:
    jobs:
      - id: one
        backend: mock
        dataset: mock
        target: api
runtime:
  runs_dir: {runs_dir}
"""


class ConfigTest(unittest.TestCase):
    def write(self, text: str, root: Path) -> Path:
        path = root / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_strict_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(self.write(BASE.format(runs_dir=root / "runs"), root))
            self.assertEqual(config.target("api").max_concurrency, 2)
            self.assertEqual(config.suite("test").jobs[0].backend, "mock")

    def test_rejects_literal_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = BASE.replace(
                "max_concurrency: 2", "max_concurrency: 2\n    api_key: secret"
            )
            with self.assertRaisesRegex(ConfigError, "unknown field.*api_key"):
                load_config(self.write(text.format(runs_dir=root / "runs"), root))

    def test_optional_key_is_resolved_only_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = BASE.replace(
                "max_concurrency: 2",
                "max_concurrency: 2\n    api_key_env: TEST_EVAL_KEY",
            )
            config = load_config(self.write(text.format(runs_dir=root / "runs"), root))
            old = os.environ.pop("TEST_EVAL_KEY", None)
            try:
                with self.assertRaisesRegex(ConfigError, "TEST_EVAL_KEY"):
                    resolve_target(config.target("api"))
                os.environ["TEST_EVAL_KEY"] = "sensitive-value"
                self.assertEqual(
                    resolve_target(config.target("api")).api_key, "sensitive-value"
                )
            finally:
                os.environ.pop("TEST_EVAL_KEY", None)
                if old is not None:
                    os.environ["TEST_EVAL_KEY"] = old


if __name__ == "__main__":
    unittest.main()
