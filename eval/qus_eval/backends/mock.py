from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ConfigError, JobConfig
from ..events import EventSink, RunEvent
from ..result import DatasetResult, ResultCounts
from ..secrets import ResolvedTarget
from .base import BackendRun, RunContext, WorkPlan


class MockBackend:
    name = "mock"

    _ALLOWED = {"items", "sleep_seconds", "wrong_every", "unknown_total", "fail_at"}

    def validate(
        self, job: JobConfig, target: ResolvedTarget | None, for_run: bool = False
    ) -> None:
        del target, for_run
        unknown = sorted(set(job.backend_args) - self._ALLOWED)
        if unknown:
            raise ConfigError(
                f"mock job {job.id} has unknown backend_args: {', '.join(unknown)}"
            )
        items = job.backend_args.get("items", 10)
        if isinstance(items, bool) or not isinstance(items, int) or items <= 0:
            raise ConfigError(
                f"mock job {job.id} backend_args.items must be a positive integer"
            )

    def plan(self, job: JobConfig, target: ResolvedTarget | None) -> WorkPlan:
        del target
        total = int(job.backend_args.get("items", 10)) * job.repeats
        if isinstance(job.limit, int):
            total = min(total, job.limit)
        elif isinstance(job.limit, float):
            total = max(1, int(total * job.limit))
        return WorkPlan(
            total=None if job.backend_args.get("unknown_total", False) else total,
            unit="samples",
            supports_resume=True,
        )

    def run(self, context: RunContext, events: EventSink) -> BackendRun:
        start = time.monotonic()
        total = int(context.job.backend_args.get("items", 10)) * context.job.repeats
        if isinstance(context.job.limit, int):
            total = min(total, context.job.limit)
        elif isinstance(context.job.limit, float):
            total = max(1, int(total * context.job.limit))
        delay = float(context.job.backend_args.get("sleep_seconds", 0.0))
        wrong_every = int(context.job.backend_args.get("wrong_every", 0))
        fail_at = context.job.backend_args.get("fail_at")
        lock = threading.Lock()
        completed = 0
        max_in_flight = 0
        in_flight = 0
        rows: list[dict] = []

        def evaluate(index: int) -> dict:
            nonlocal in_flight, max_in_flight
            if context.cancel_event.is_set():
                raise InterruptedError("run cancelled")
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                if delay:
                    time.sleep(delay)
                if fail_at is not None and index == int(fail_at):
                    raise RuntimeError(f"configured mock failure at item {index}")
                return {
                    "index": index,
                    "correct": not wrong_every or (index + 1) % wrong_every != 0,
                }
            finally:
                with lock:
                    in_flight -= 1

        with ThreadPoolExecutor(max_workers=context.granted_concurrency) as pool:
            futures = {pool.submit(evaluate, index): index for index in range(total)}
            for future in as_completed(futures):
                if context.cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    raise InterruptedError("run cancelled")
                row = future.result()
                rows.append(row)
                completed += 1
                events.emit(
                    RunEvent(
                        kind="progress",
                        run_id=context.run_id,
                        job_id=context.job.id,
                        phase="inference",
                        completed=completed,
                        total=context.plan.total,
                        unit="samples",
                        message="mock evaluation",
                    )
                )
        rows.sort(key=lambda item: item["index"])
        raw = {"rows": rows, "max_in_flight": max_in_flight}
        artifact = context.job_dir / "mock-result.json"
        artifact.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        return BackendRun(start, time.monotonic(), raw, [artifact])

    def normalize(self, context: RunContext, run: BackendRun) -> DatasetResult:
        rows = run.raw_result["rows"]
        correct = sum(bool(row["correct"]) for row in rows)
        total = len(rows)
        return DatasetResult(
            job_id=context.job.id,
            backend=self.name,
            dataset=context.job.dataset,
            status="completed",
            primary_metric="accuracy",
            metrics={
                "accuracy": correct / total if total else 0.0,
                "max_in_flight": run.raw_result["max_in_flight"],
            },
            counts=ResultCounts(
                planned=context.plan.total, completed=total, scored=total
            ),
            duration_seconds=run.duration_seconds,
            artifacts=[
                str(path.relative_to(context.job_dir.parent.parent))
                for path in run.artifacts
            ],
        )
