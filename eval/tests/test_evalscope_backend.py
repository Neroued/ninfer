import json
import tempfile
import threading
import time
import unittest
import os
from pathlib import Path

from qus_eval.backends.base import BackendRun, RunContext, WorkPlan
from qus_eval.backends.evalscope import EvalScopeBackend
from qus_eval.config import JobConfig


class EvalScopeBackendTest(unittest.TestCase):
    def test_normalizes_accuracy_without_importing_evalscope(self):
        backend = EvalScopeBackend()
        job = JobConfig("gpqa", "evalscope", "gpqa_diamond", "api", 1, None, 1, {}, {})
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "backends" / "gpqa"
            job_dir.mkdir(parents=True)
            context = RunContext(
                "run", job, job_dir, None, 1, threading.Event(), False, WorkPlan(198)
            )
            raw = {"gpqa_diamond": {"score": 0.625, "num": 198, "metrics": []}}
            result = backend.normalize(
                context, BackendRun(time.monotonic(), time.monotonic(), raw)
            )
            self.assertEqual(result.primary_metric, "accuracy")
            self.assertEqual(result.metrics["accuracy"], 0.625)
            self.assertEqual(result.counts.completed, 198)

    def test_normalizes_bfcl_official_aggregates(self):
        backend = EvalScopeBackend()
        job = JobConfig("bfcl", "evalscope", "bfcl_v4", "api", 1, None, 1, {}, {})
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "backends" / "bfcl"
            job_dir.mkdir(parents=True)
            context = RunContext(
                "run", job, job_dir, None, 1, threading.Event(), False, WorkPlan(5106)
            )
            subsets = [
                {"name": "AGENTIC", "score": 0.4, "num": 1},
                {"name": "MULTI_TURN", "score": 0.5, "num": 1},
                {"name": "OVERALL", "score": 0.45, "num": 1},
            ]
            raw = {
                "bfcl_v4": {
                    "score": 0.45,
                    "num": 5106,
                    "metrics": [{"categories": [{"subsets": subsets}]}],
                }
            }
            result = backend.normalize(
                context, BackendRun(time.monotonic(), time.monotonic(), raw)
            )
            self.assertEqual(result.primary_metric, "overall")
            self.assertEqual(result.metrics["overall"], 0.45)
            self.assertEqual(result.metrics["agentic"], 0.4)

    def test_partial_bfcl_uses_selected_sample_accuracy(self):
        backend = EvalScopeBackend()
        job = JobConfig(
            "bfcl",
            "evalscope",
            "bfcl_v4",
            "api",
            1,
            2,
            1,
            {},
            {"subset_list": ["simple_python", "multi_turn_base"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "backends" / "bfcl"
            job_dir.mkdir(parents=True)
            context = RunContext(
                "run", job, job_dir, None, 1, threading.Event(), False, WorkPlan(4)
            )
            raw = {
                "bfcl_v4": {
                    "score": 1.0,
                    "num": 4,
                    "metrics": [
                        {
                            "categories": [
                                {
                                    "subsets": [
                                        {"name": "MULTI_TURN", "score": 1.0},
                                        {"name": "OVERALL", "score": 0.5},
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }
            result = backend.normalize(
                context, BackendRun(time.monotonic(), time.monotonic(), raw)
            )
            self.assertEqual(result.primary_metric, "accuracy")
            self.assertEqual(result.metrics["accuracy"], 1.0)
            self.assertEqual(result.metrics["multi_turn"], 1.0)
            self.assertNotIn("overall", result.metrics)

    def test_normalization_surfaces_provider_errors_wrapped_by_bfcl(self):
        backend = EvalScopeBackend()
        job = JobConfig(
            "bfcl",
            "evalscope",
            "bfcl_v4",
            "api",
            1,
            1,
            1,
            {},
            {"subset_list": ["simple_python"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "backends" / "bfcl"
            predictions = job_dir / "predictions" / "model"
            predictions.mkdir(parents=True)
            wrapped = {
                "error": "Error code: 400",
                "error_message": "provider traceback",
            }
            record = {
                "model_output": {
                    "error": None,
                    "choices": [{"message": {"content": json.dumps(wrapped)}}],
                }
            }
            (predictions / "bfcl.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            context = RunContext(
                "run", job, job_dir, None, 1, threading.Event(), False, WorkPlan(1)
            )
            raw = {"bfcl_v4": {"score": 0.0, "num": 1, "metrics": []}}
            result = backend.normalize(
                context, BackendRun(time.monotonic(), time.monotonic(), raw)
            )
            self.assertEqual(result.counts.completed, 1)
            self.assertEqual(result.counts.failed, 1)

    def test_bfcl_task_snapshot_does_not_contain_serpapi_secret(self):
        backend = EvalScopeBackend()
        job = JobConfig(
            "bfcl",
            "evalscope",
            "bfcl_v4",
            "api",
            1,
            1,
            1,
            {},
            {
                "subset_list": ["web_search_base"],
                "serpapi_api_key_env": "TEST_SERP_KEY",
            },
        )
        old = os.environ.get("TEST_SERP_KEY")
        os.environ["TEST_SERP_KEY"] = "secret-serp-value"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                from qus_eval.config import RequestConfig, TargetConfig
                from qus_eval.secrets import ResolvedTarget

                target = ResolvedTarget(
                    TargetConfig(
                        "api",
                        "openai_chat",
                        "http://localhost/v1",
                        "m",
                        None,
                        1,
                        RequestConfig(),
                    ),
                    None,
                )
                context = RunContext(
                    "run",
                    job,
                    Path(tmp),
                    target,
                    1,
                    threading.Event(),
                    False,
                    backend.plan(job, target),
                )
                encoded = str(backend._task_dict(context))
                self.assertNotIn("secret-serp-value", encoded)
                with backend._bfcl_environment(job):
                    self.assertEqual(os.environ["SERPAPI_API_KEY"], "secret-serp-value")
        finally:
            os.environ.pop("SERPAPI_API_KEY", None)
            if old is None:
                os.environ.pop("TEST_SERP_KEY", None)
            else:
                os.environ["TEST_SERP_KEY"] = old

    def test_none_sample_retention_removes_prediction_and_review_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dirname in ("predictions", "reviews"):
                directory = root / dirname
                directory.mkdir()
                (directory / "samples.jsonl").write_text('{"sample": 1}\n')
            EvalScopeBackend._apply_sample_retention(root, "none")
            self.assertFalse((root / "predictions").exists())
            self.assertFalse((root / "reviews").exists())


if __name__ == "__main__":
    unittest.main()
