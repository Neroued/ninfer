import io
import logging
import tempfile
import unittest
from pathlib import Path

from ninfer_eval.events import EventSink, RunEvent
from ninfer_eval.progress import ProgressRenderer


class NonTTY(io.StringIO):
    def isatty(self):
        return False


class ProgressTest(unittest.TestCase):
    def test_non_tty_prints_lifecycle_and_does_not_invent_total(self):
        stream = NonTTY()
        renderer = ProgressRenderer(enabled=True, heartbeat_seconds=60, stream=stream)
        renderer.handle(
            RunEvent(
                kind="job_start", run_id="r", job_id="j", phase="planning", completed=0
            )
        )
        renderer.handle(
            RunEvent(
                kind="progress", run_id="r", job_id="j", phase="inference", completed=1
            )
        )
        renderer.handle(
            RunEvent(
                kind="job_end", run_id="r", job_id="j", phase="reporting", completed=2
            )
        )
        renderer.close()
        output = stream.getvalue()
        self.assertIn("0/?", output)
        self.assertIn("2/?", output)
        self.assertNotIn("%", output)

    def test_event_artifacts_redact_known_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logger = logging.getLogger(f"redaction-{id(self)}")
            logger.handlers.clear()
            logger.propagate = False
            handler = logging.FileHandler(root / "run.log")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            sink = EventSink(root / "events.jsonl", logger, secrets=["top-secret"])
            sink.emit(RunEvent(kind="warning", run_id="r", message="token=top-secret"))
            sink.close()
            handler.close()
            self.assertNotIn("top-secret", (root / "events.jsonl").read_text())
            self.assertNotIn("top-secret", (root / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()
