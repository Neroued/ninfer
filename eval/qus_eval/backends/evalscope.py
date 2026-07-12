from __future__ import annotations

import importlib.util
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import ConfigError, JobConfig
from ..events import EventSink, RunEvent
from ..result import DatasetResult, ResultCounts
from ..secrets import ResolvedTarget
from .base import BackendDependencyError, BackendRun, RunContext, WorkPlan

_DATASET_COUNTS = {
    "aime25": {"default": 30},
    "aime26": {"default": 30},
    "gpqa_diamond": {"default": 198},
    "bfcl_v4": {
        "simple_python": 400,
        "simple_java": 100,
        "simple_javascript": 50,
        "multiple": 200,
        "parallel": 200,
        "parallel_multiple": 200,
        "irrelevance": 240,
        "live_simple": 258,
        "live_multiple": 1053,
        "live_parallel": 16,
        "live_parallel_multiple": 24,
        "live_irrelevance": 884,
        "live_relevance": 16,
        "multi_turn_base": 200,
        "multi_turn_miss_func": 200,
        "multi_turn_miss_param": 200,
        "multi_turn_long_context": 200,
        "web_search_base": 100,
        "web_search_no_snippet": 100,
        "memory_kv": 155,
        "memory_vector": 155,
        "memory_rec_sum": 155,
    },
}

_BFCL_AGGREGATES = {
    "agentic",
    "multi_turn",
    "non_live",
    "live",
    "hallucination",
    "overall",
}


class EvalScopeBackend:
    name = "evalscope"
    _serpapi_lock = threading.Lock()

    _ALLOWED_ARGS = {
        "subset_list",
        "dataset_args",
        "few_shot_num",
        "shuffle",
        "judge_strategy",
        "judge_model_args",
        "dataset_hub",
        "dataset_dir",
        "collect_perf",
        "ignore_errors",
        "debug",
        "is_fc_model",
        "underscore_to_dot",
        "serpapi_api_key_env",
        "allow_network_downloads",
    }

    def validate(
        self, job: JobConfig, target: ResolvedTarget | None, for_run: bool = False
    ) -> None:
        if target is None:
            raise ConfigError(f"EvalScope job {job.id} requires a target")
        if target.config.protocol != "openai_chat":
            raise ConfigError(
                f"EvalScope job {job.id} currently supports protocol openai_chat, got {target.config.protocol}"
            )
        unknown = sorted(set(job.backend_args) - self._ALLOWED_ARGS)
        if unknown:
            raise ConfigError(
                f"EvalScope job {job.id} has unknown backend_args: {', '.join(unknown)}"
            )
        subsets = job.backend_args.get("subset_list")
        if subsets is not None and (
            not isinstance(subsets, list)
            or not subsets
            or not all(isinstance(value, str) for value in subsets)
        ):
            raise ConfigError(
                f"EvalScope job {job.id} subset_list must be a non-empty string list"
            )
        if job.dataset == "bfcl_v4":
            self._validate_bfcl(job, for_run)
        if for_run:
            self._require_package("evalscope", "1.9.0")

    def _validate_bfcl(self, job: JobConfig, for_run: bool) -> None:
        subsets = set(job.backend_args.get("subset_list") or _DATASET_COUNTS["bfcl_v4"])
        unknown = sorted(subsets - set(_DATASET_COUNTS["bfcl_v4"]))
        if unknown:
            raise ConfigError(
                f"BFCL-v4 job {job.id} has unknown subsets: {', '.join(unknown)}"
            )
        if for_run:
            self._require_package("bfcl-eval", "2025.10.27.1")
            self._require_package("soundfile", "0.14.0")
            web = {"web_search_base", "web_search_no_snippet"}
            env_name = job.backend_args.get("serpapi_api_key_env")
            if subsets & web and (not env_name or not os.environ.get(str(env_name))):
                raise ConfigError(
                    f"BFCL-v4 job {job.id} includes web-search subsets and requires "
                    "backend_args.serpapi_api_key_env pointing to a populated environment variable"
                )
            if "memory_vector" in subsets and not job.backend_args.get(
                "allow_network_downloads", False
            ):
                raise ConfigError(
                    f"BFCL-v4 job {job.id} includes memory_vector; set "
                    "backend_args.allow_network_downloads=true to acknowledge its model download"
                )

    @staticmethod
    def _require_package(package: str, expected: str) -> None:
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise BackendDependencyError(
                f"required package is not installed: {package}=={expected}"
            ) from exc
        if installed != expected:
            raise BackendDependencyError(
                f"{package} version mismatch: installed={installed}, required={expected}"
            )

    def plan(self, job: JobConfig, target: ResolvedTarget | None) -> WorkPlan:
        del target
        counts = _DATASET_COUNTS.get(job.dataset)
        subsets = tuple(
            job.backend_args.get("subset_list") or (counts.keys() if counts else ())
        )
        total = None
        if counts:
            total = 0
            for subset in subsets:
                count = counts.get(subset)
                if count is None:
                    continue
                if isinstance(job.limit, int):
                    count = min(count, job.limit)
                elif isinstance(job.limit, float):
                    count = max(1, int(count * job.limit))
                total += count * job.repeats
        requirements = ["evalscope==1.9.0"]
        warnings: list[str] = []
        if job.dataset == "bfcl_v4":
            requirements.append("bfcl-eval==2025.10.27.1")
            requirements.append("soundfile==0.14.0")
            if {"web_search_base", "web_search_no_snippet"} & set(subsets):
                requirements.append("SerpAPI key")
            if "memory_vector" in subsets:
                requirements.append("network access for the BFCL memory-vector model")
            warnings.append(
                "BFCL multi-turn samples may issue more than one model request"
            )
        return WorkPlan(
            total=total,
            unit="samples",
            subsets=subsets,
            supports_resume=True,
            requirements=tuple(requirements),
            warnings=tuple(warnings),
        )

    def run(self, context: RunContext, events: EventSink) -> BackendRun:
        if importlib.util.find_spec("evalscope") is None:
            raise RuntimeError("EvalScope is not installed")
        from evalscope import TaskConfig, run_task

        task_dict = self._task_dict(context)
        task_path = context.job_dir / "evalscope-task.json"
        safe_task = dict(task_dict)
        if safe_task.get("api_key") not in {None, "EMPTY"}:
            safe_task["api_key"] = "<redacted>"
        task_path.write_text(
            json.dumps(safe_task, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

        events.emit(
            RunEvent(
                kind="phase",
                run_id=context.run_id,
                job_id=context.job.id,
                phase="loading",
                completed=0,
                total=context.plan.total,
                unit=context.plan.unit,
                message="loading EvalScope benchmark",
            )
        )
        stop_poll = threading.Event()
        run_started_wall = time.time()
        poller = threading.Thread(
            target=self._poll_progress,
            args=(context, events, stop_poll, run_started_wall),
            name=f"evalscope-progress-{context.job.id}",
            daemon=True,
        )
        poller.start()
        start = time.monotonic()
        try:
            with self._bfcl_environment(context.job):
                raw = run_task(TaskConfig(**task_dict))
        finally:
            stop_poll.set()
            poller.join(timeout=5)
        finish = time.monotonic()
        raw_path = context.job_dir / "evalscope-return.json"
        raw_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        artifacts = [task_path, raw_path]
        report = self._find_report_file(context.job_dir, context.job.dataset)
        if report:
            artifacts.append(report)
        self._apply_sample_retention(context.job_dir, context.sample_retention)
        events.emit(
            RunEvent(
                kind="phase",
                run_id=context.run_id,
                job_id=context.job.id,
                phase="reporting",
                completed=context.plan.total,
                total=context.plan.total,
                unit=context.plan.unit,
                message="normalizing EvalScope report",
            )
        )
        return BackendRun(start, finish, raw, artifacts)

    def _task_dict(self, context: RunContext) -> dict[str, Any]:
        assert context.target is not None
        job = context.job
        args = job.backend_args
        dataset_args = dict(args.get("dataset_args", {}))
        if args.get("subset_list") is not None:
            dataset_args["subset_list"] = args["subset_list"]
        if args.get("few_shot_num") is not None:
            dataset_args["few_shot_num"] = args["few_shot_num"]
        if args.get("shuffle") is not None:
            dataset_args["shuffle"] = args["shuffle"]
        if job.dataset == "bfcl_v4":
            extra = dict(dataset_args.get("extra_params", {}))
            extra["is_fc_model"] = bool(args.get("is_fc_model", True))
            extra["underscore_to_dot"] = bool(args.get("underscore_to_dot", True))
            dataset_args["extra_params"] = extra

        generation = dict(job.generation)
        generation.setdefault("timeout", context.target.config.request.timeout_seconds)
        generation.setdefault("retries", context.target.config.request.retries)
        generation.setdefault(
            "retry_interval", context.target.config.request.retry_interval_seconds
        )
        if context.target.config.request.headers:
            generation.setdefault(
                "extra_headers", context.target.config.request.headers
            )
        seed = generation.get("seed", 42)
        task: dict[str, Any] = {
            "model": context.target.config.model,
            "model_id": context.target.config.model,
            "api_url": context.target.config.base_url,
            "api_key": context.target.api_key or "EMPTY",
            "eval_type": "openai_api",
            "datasets": [job.dataset],
            "dataset_args": {job.dataset: dataset_args},
            "eval_batch_size": context.granted_concurrency,
            "generation_config": generation,
            "limit": job.limit,
            "repeats": job.repeats,
            "seed": seed,
            "judge_strategy": args.get("judge_strategy", "auto"),
            "judge_model_args": args.get("judge_model_args", {}),
            "collect_perf": bool(args.get("collect_perf", True)),
            "ignore_errors": bool(args.get("ignore_errors", False)),
            "debug": bool(args.get("debug", False)),
            "enable_progress_tracker": True,
            "work_dir": str(context.job_dir),
            "no_timestamp": True,
        }
        if args.get("dataset_hub") is not None:
            task["dataset_hub"] = args["dataset_hub"]
        if args.get("dataset_dir") is not None:
            task["dataset_dir"] = args["dataset_dir"]
        if context.resume and (context.job_dir / "predictions").exists():
            task["use_cache"] = str(context.job_dir)
        return task

    @staticmethod
    def _poll_progress(
        context: RunContext,
        events: EventSink,
        stop: threading.Event,
        started_wall: float,
    ) -> None:
        path = context.job_dir / "progress.json"
        last: tuple[Any, ...] | None = None
        while not stop.wait(0.5):
            if context.cancel_event.is_set():
                return
            try:
                if path.stat().st_mtime < started_wall:
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                key = (
                    data.get("processed_count"),
                    data.get("total_count"),
                    data.get("status"),
                )
                if key == last:
                    continue
                last = key
                events.emit(
                    RunEvent(
                        kind="progress",
                        run_id=context.run_id,
                        job_id=context.job.id,
                        phase="inference",
                        completed=data.get("processed_count"),
                        total=data.get("total_count"),
                        unit="samples",
                        message=f"EvalScope {data.get('status', 'running')}",
                    )
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue

    def normalize(self, context: RunContext, run: BackendRun) -> DatasetResult:
        report = self._report_dict(context, run.raw_result)
        score = report.get("score")
        num = int(report.get("num") or 0)
        metrics: dict[str, Any] = {}
        primary = "accuracy"
        if context.job.dataset == "bfcl_v4":
            metrics.update(self._bfcl_metrics(report))
            full_subsets = set(
                context.job.backend_args.get("subset_list")
                or _DATASET_COUNTS["bfcl_v4"]
            )
            is_formal_full = (
                full_subsets == set(_DATASET_COUNTS["bfcl_v4"])
                and context.job.limit is None
                and context.job.repeats == 1
            )
            if is_formal_full:
                if "overall" not in metrics:
                    raise RuntimeError(
                        "full BFCL-v4 run did not produce the official OVERALL score"
                    )
                score = metrics["overall"]
                primary = "overall"
            else:
                # BFCL still emits an OVERALL aggregate for partial subset runs, but
                # its fixed category weighting includes categories absent from the
                # run. Use the report's selected-sample accuracy for smoke/custom
                # subsets and reserve official OVERALL for the complete benchmark.
                metrics.pop("overall", None)
        if primary == "accuracy":
            metrics["accuracy"] = (
                float(score) if isinstance(score, (int, float)) else 0.0
            )
        report_path = self._find_report_file(context.job_dir, context.job.dataset)
        artifacts = [
            str(path.relative_to(context.job_dir.parent.parent))
            for path in run.artifacts
            if path.exists()
        ]
        if (
            report_path
            and str(report_path.relative_to(context.job_dir.parent.parent))
            not in artifacts
        ):
            artifacts.append(
                str(report_path.relative_to(context.job_dir.parent.parent))
            )
        failed = self._prediction_failures(context.job_dir)
        return DatasetResult(
            job_id=context.job.id,
            backend=self.name,
            dataset=context.job.dataset,
            status="completed",
            primary_metric=primary,
            metrics=metrics,
            counts=ResultCounts(
                planned=context.plan.total,
                completed=num,
                scored=num,
                failed=failed,
            ),
            duration_seconds=run.duration_seconds,
            artifacts=artifacts,
        )

    @staticmethod
    def _prediction_failures(job_dir: Path) -> int:
        """Count EvalScope records that wrapped an inference exception as output.

        BFCL catches provider errors and serializes them into an assistant message,
        so its report still counts the sample and the outer task still succeeds.
        Audit the retained predictions to keep those infrastructure failures visible
        in the normalized result instead of silently treating them as model mistakes.
        """
        failures = 0
        for path in (job_dir / "predictions").glob("**/*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    output = record.get("model_output") or {}
                    if output.get("error") is not None:
                        failures += 1
                        continue
                    choices = output.get("choices") or []
                    content = choices[0]["message"]["content"] if choices else None
                    wrapped = json.loads(content) if isinstance(content, str) else None
                    if (
                        isinstance(wrapped, dict)
                        and {
                            "error",
                            "error_message",
                        }
                        <= wrapped.keys()
                    ):
                        failures += 1
                except (json.JSONDecodeError, KeyError, TypeError, OSError):
                    continue
        return failures

    def _report_dict(self, context: RunContext, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            candidate = raw.get(context.job.dataset)
            if isinstance(candidate, dict) and (
                "score" in candidate or "metrics" in candidate
            ):
                return candidate
            if raw.get("dataset_name") == context.job.dataset:
                return raw
        report_path = self._find_report_file(context.job_dir, context.job.dataset)
        if report_path is None:
            raise RuntimeError(
                f"EvalScope did not produce a report for {context.job.dataset}"
            )
        return json.loads(report_path.read_text(encoding="utf-8"))

    @staticmethod
    def _find_report_file(root: Path, dataset: str) -> Path | None:
        candidates = sorted(root.glob(f"reports/**/{dataset}.json"))
        return candidates[0] if candidates else None

    @staticmethod
    def _bfcl_metrics(report: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        for metric in report.get("metrics", []):
            for category in metric.get("categories", []):
                for subset in category.get("subsets", []):
                    name = str(subset.get("name", "")).lower()
                    if name in _BFCL_AGGREGATES and isinstance(
                        subset.get("score"), (int, float)
                    ):
                        values[name] = float(subset["score"])
        return values

    @classmethod
    @contextmanager
    def _bfcl_environment(cls, job: JobConfig):
        env_name = (
            job.backend_args.get("serpapi_api_key_env")
            if job.dataset == "bfcl_v4"
            else None
        )
        if not env_name:
            yield
            return
        value = os.environ.get(str(env_name))
        with cls._serpapi_lock:
            old = os.environ.get("SERPAPI_API_KEY")
            if value:
                os.environ["SERPAPI_API_KEY"] = value
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop("SERPAPI_API_KEY", None)
                else:
                    os.environ["SERPAPI_API_KEY"] = old

    @staticmethod
    def _apply_sample_retention(job_dir: Path, policy: str) -> None:
        if policy == "all":
            return
        for dirname in ("predictions", "reviews"):
            root = job_dir / dirname
            if not root.exists():
                continue
            if policy == "none":
                shutil.rmtree(root)
                continue
            for path in root.rglob("*.jsonl"):
                kept: list[str] = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        kept.append(line)
                        continue
                    encoded = json.dumps(record, ensure_ascii=False).lower()
                    if any(
                        marker in encoded
                        for marker in (
                            '"error"',
                            "error_message",
                            "traceback",
                            "exception",
                        )
                    ):
                        kept.append(line)
                if kept:
                    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
                else:
                    path.unlink()
