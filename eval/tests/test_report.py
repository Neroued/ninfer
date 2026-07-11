import json
import tempfile
import unittest
from pathlib import Path

from qus_eval.result import DatasetResult, ResultCounts, write_summary


class ReportTest(unittest.TestCase):
    def test_writes_stable_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = DatasetResult(
                job_id="aime25",
                backend="mock",
                dataset="aime25",
                status="completed",
                primary_metric="accuracy",
                metrics={"accuracy": 0.5},
                counts=ResultCounts(planned=30, completed=30, scored=30),
                duration_seconds=12.5,
            )
            write_summary(root, "run", "completed", [result])
            payload = json.loads((root / "summary.json").read_text())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["results"][0]["counts"]["planned"], 30)
            self.assertIn(
                "| aime25 | mock | aime25 |", (root / "summary.md").read_text()
            )


if __name__ == "__main__":
    unittest.main()
