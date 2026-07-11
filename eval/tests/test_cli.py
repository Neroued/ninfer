import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from qus_eval.cli import EXIT_CONFIG, main


class CliTest(unittest.TestCase):
    def test_configuration_error_is_reported_to_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text("schema_version: 2\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main(["validate", "--config", str(path)])
            self.assertEqual(status, EXIT_CONFIG)
            self.assertIn("configuration error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
