import json
import tempfile
import unittest
from pathlib import Path

from qus_eval.config import load_config
from qus_eval.coordinator import Coordinator, load_resume_config


class CoordinatorTest(unittest.TestCase):
    def make_config(self, root: Path) -> Path:
        path = root / "suite.yaml"
        path.write_text(
            f"""
schema_version: 1
targets:
  api:
    protocol: openai_chat
    base_url: http://127.0.0.1:8000/v1
    model: mock
    max_concurrency: 3
suites:
  all:
    jobs:
      - id: first
        backend: mock
        dataset: first
        target: api
        max_concurrency: 2
        backend_args:
          items: 8
          sleep_seconds: 0.02
          wrong_every: 4
      - id: second
        backend: mock
        dataset: second
        target: api
        max_concurrency: 2
        backend_args:
          items: 8
          sleep_seconds: 0.02
runtime:
  max_parallel_jobs: 2
  runs_dir: {root / 'runs'}
  progress:
    enabled: false
""",
            encoding="utf-8",
        )
        return path

    def test_run_enforces_target_capacity_and_writes_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(self.make_config(root))
            run_dir = Coordinator(config, "all").run()
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(
                [r["metrics"]["accuracy"] for r in summary["results"]], [0.75, 1.0]
            )
            # Each backend reports its observed internal peak. Slot reservations ensure the sum
            # across concurrently active jobs cannot exceed the target capacity of three.
            observed = [r["metrics"]["max_in_flight"] for r in summary["results"]]
            self.assertLessEqual(sum(observed), 3)
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "events.jsonl").exists())
            self.assertTrue((run_dir / "summary.md").exists())

    def test_resume_skips_completed_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(self.make_config(root))
            run_dir = Coordinator(config, "all").run()
            resumed_config, suite = load_resume_config(run_dir)
            resumed = Coordinator(resumed_config, suite).run(resume_dir=run_dir)
            self.assertEqual(resumed, run_dir)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            skipped = {
                event["job_id"] for event in events if event["kind"] == "job_skipped"
            }
            self.assertEqual(skipped, {"first", "second"})


if __name__ == "__main__":
    unittest.main()
