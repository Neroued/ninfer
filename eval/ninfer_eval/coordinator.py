from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import __version__
from .backends.base import RunContext
from .backends.registry import get_backend
from .config import AppConfig, ConfigError, JobConfig, dump_config, load_config
from .events import EventSink, RunEvent
from .logging import close_logger, create_run_logger
from .progress import ProgressRenderer
from .result import DatasetResult, ResultCounts, _atomic_json, write_summary
from .secrets import ResolvedTarget, resolve_target


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TargetReservations:
    def __init__(self, capacities: dict[str, int]):
        self._capacities = capacities
        self._available = dict(capacities)
        self._condition = threading.Condition()

    def acquire(
        self, target: str | None, requested: int, cancel: threading.Event
    ) -> int:
        if target is None:
            return requested
        with self._condition:
            while self._available[target] <= 0:
                if cancel.is_set():
                    raise InterruptedError(
                        "run cancelled while waiting for target capacity"
                    )
                self._condition.wait(timeout=0.2)
            grant = min(requested, self._available[target])
            self._available[target] -= grant
            return grant

    def release(self, target: str | None, granted: int) -> None:
        if target is None:
            return
        with self._condition:
            self._available[target] += granted
            if self._available[target] > self._capacities[target]:
                raise RuntimeError(f"released too many slots for target {target}")
            self._condition.notify_all()


class RunState:
    def __init__(
        self,
        path: Path,
        run_id: str,
        jobs: tuple[JobConfig, ...],
        existing: dict | None = None,
    ):
        self.path = path
        self._lock = threading.Lock()
        self.data = existing or {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "jobs": {job.id: {"status": "pending"} for job in jobs},
        }
        self.write()

    def update_job(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self.data["jobs"].setdefault(job_id, {}).update(values)
            self._write_unlocked()

    def update_run(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values)
            self._write_unlocked()

    def write(self) -> None:
        with self._lock:
            self._write_unlocked()

    def _write_unlocked(self) -> None:
        _atomic_json(self.path, self.data)


def validate_suite(config: AppConfig, suite_name: str, for_run: bool = False) -> None:
    suite = config.suite(suite_name)
    for job in suite.jobs:
        target_cfg = config.target(job.target)
        target = (
            resolve_target(target_cfg)
            if for_run
            else (ResolvedTarget(target_cfg, None) if target_cfg is not None else None)
        )
        backend = get_backend(job.backend)
        backend.validate(job, target, for_run=for_run)
        if (
            target_cfg
            and job.max_concurrency
            and job.max_concurrency > target_cfg.max_concurrency
        ):
            raise ConfigError(
                f"job {job.id} max_concurrency {job.max_concurrency} exceeds target "
                f"{target_cfg.name} capacity {target_cfg.max_concurrency}"
            )


def plan_suite(
    config: AppConfig, suite_name: str, for_run: bool = False
) -> list[dict[str, Any]]:
    validate_suite(config, suite_name, for_run=for_run)
    plans = []
    for job in config.suite(suite_name).jobs:
        target_cfg = config.target(job.target)
        target = (
            resolve_target(target_cfg)
            if for_run
            else (ResolvedTarget(target_cfg, None) if target_cfg is not None else None)
        )
        plan = get_backend(job.backend).plan(job, target)
        requested = job.max_concurrency or (
            target_cfg.max_concurrency if target_cfg else 1
        )
        plans.append(
            {
                "job_id": job.id,
                "backend": job.backend,
                "dataset": job.dataset,
                "target": job.target,
                "requested_concurrency": requested,
                "total": plan.total,
                "unit": plan.unit,
                "subsets": list(plan.subsets),
                "supports_resume": plan.supports_resume,
                "requirements": list(plan.requirements),
                "warnings": list(plan.warnings),
            }
        )
    return plans


class Coordinator:
    def __init__(
        self,
        config: AppConfig,
        suite_name: str,
        cancel_event: threading.Event | None = None,
    ):
        self.config = config
        self.suite = config.suite(suite_name)
        self.suite_name = suite_name
        self.cancel_event = cancel_event or threading.Event()

    def run(self, resume_dir: Path | None = None) -> Path:
        validate_suite(self.config, self.suite_name, for_run=True)
        fingerprint = self.config.fingerprint(self.suite_name)
        if resume_dir is None:
            run_id = f"{_utc_compact()}-{fingerprint[:8]}"
            run_dir = Path(self.config.runtime.runs_dir).resolve() / run_id
            if run_dir.exists():
                suffix = 1
                while (run_dir.parent / f"{run_id}-{suffix}").exists():
                    suffix += 1
                run_dir = run_dir.parent / f"{run_id}-{suffix}"
                run_id = run_dir.name
            run_dir.mkdir(parents=True)
            dump_config(self.config, run_dir / "effective-config.yaml")
            state_existing = None
            manifest = self._make_manifest(run_id, fingerprint)
            _atomic_json(run_dir / "manifest.json", manifest)
        else:
            run_dir = resume_dir.resolve()
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("config_fingerprint") != fingerprint:
                raise ConfigError(
                    "resume configuration fingerprint mismatch: "
                    f"stored={manifest.get('config_fingerprint')} current={fingerprint}"
                )
            run_id = manifest["run_id"]
            state_existing = json.loads(
                (run_dir / "state.json").read_text(encoding="utf-8")
            )
            state_existing["status"] = "running"

        jobs_dir = run_dir / "backends"
        jobs_dir.mkdir(exist_ok=True)
        logger = create_run_logger(run_dir)
        used_target_names = {
            job.target for job in self.suite.jobs if job.target is not None
        }
        resolved_targets = {
            name: resolve_target(self.config.targets[name])
            for name in used_target_names
        }
        secrets = [
            target.api_key
            for target in resolved_targets.values()
            if target and target.api_key
        ]
        secrets.extend(_configured_backend_secrets(self.suite.jobs))
        renderer = ProgressRenderer(
            enabled=self.config.runtime.progress.enabled,
            heartbeat_seconds=self.config.runtime.progress.heartbeat_seconds,
            refresh_seconds=self.config.runtime.progress.refresh_seconds,
        )
        events = EventSink(run_dir / "events.jsonl", logger, renderer, secrets)
        state = RunState(
            run_dir / "state.json", run_id, self.suite.jobs, state_existing
        )
        reservations = TargetReservations(
            {
                name: target.max_concurrency
                for name, target in self.config.targets.items()
            }
        )
        events.emit(
            RunEvent(
                kind="run_start",
                run_id=run_id,
                completed=0,
                total=len(self.suite.jobs),
                unit="jobs",
                message=f"suite={self.suite_name}",
            )
        )

        results_by_id: dict[str, DatasetResult] = {}
        runnable: list[JobConfig] = []
        for job in self.suite.jobs:
            saved = state.data["jobs"].get(job.id, {})
            result_path = jobs_dir / job.id / "job-result.json"
            if saved.get("status") == "completed" and result_path.exists():
                results_by_id[job.id] = DatasetResult.from_dict(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                events.emit(
                    RunEvent(
                        kind="job_skipped",
                        run_id=run_id,
                        job_id=job.id,
                        message="already completed",
                    )
                )
            else:
                runnable.append(job)

        try:
            with ThreadPoolExecutor(
                max_workers=self.config.runtime.max_parallel_jobs
            ) as pool:
                futures = {
                    pool.submit(
                        self._run_job,
                        job,
                        run_id,
                        jobs_dir,
                        resolved_targets.get(job.target) if job.target else None,
                        reservations,
                        state,
                        events,
                        resume_dir is not None,
                    ): job
                    for job in runnable
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        results_by_id[job.id] = future.result()
                    except InterruptedError as exc:
                        self.cancel_event.set()
                        results_by_id[job.id] = self._failed_result(
                            job, "cancelled", str(exc)
                        )
                    except Exception as exc:
                        results_by_id[job.id] = self._failed_result(
                            job, "failed", str(exc)
                        )
            ordered = [results_by_id[job.id] for job in self.suite.jobs]
            if self.cancel_event.is_set():
                final_status = "cancelled"
            elif all(result.status == "completed" for result in ordered):
                final_status = "completed"
            elif any(result.status == "completed" for result in ordered):
                final_status = "partial"
            else:
                final_status = "failed"
            write_summary(run_dir, run_id, final_status, ordered)
            state.update_run(
                status=final_status, finished_at=datetime.now(timezone.utc).isoformat()
            )
            events.emit(
                RunEvent(
                    kind="run_end",
                    run_id=run_id,
                    completed=len(ordered),
                    total=len(ordered),
                    unit="jobs",
                    message=final_status,
                )
            )
        finally:
            events.close()
            close_logger(logger)
        return run_dir

    def _run_job(
        self,
        job: JobConfig,
        run_id: str,
        jobs_dir: Path,
        target: ResolvedTarget | None,
        reservations: TargetReservations,
        state: RunState,
        events: EventSink,
        resume: bool,
    ) -> DatasetResult:
        backend = get_backend(job.backend)
        plan = backend.plan(job, target)
        target_cfg = target.config if target else None
        requested = job.max_concurrency or (
            target_cfg.max_concurrency if target_cfg else 1
        )
        granted = reservations.acquire(job.target, requested, self.cancel_event)
        job_dir = jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        context = RunContext(
            run_id=run_id,
            job=job,
            job_dir=job_dir,
            target=target,
            granted_concurrency=granted,
            cancel_event=self.cancel_event,
            resume=resume,
            plan=plan,
            sample_retention=self.config.runtime.sample_retention,
        )
        state.update_job(
            job.id,
            status="running",
            error=None,
            granted_concurrency=granted,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        events.emit(
            RunEvent(
                kind="job_start",
                run_id=run_id,
                job_id=job.id,
                phase="planning",
                completed=0,
                total=plan.total,
                unit=plan.unit,
                message=f"backend={job.backend} dataset={job.dataset} concurrency={granted}",
            )
        )
        started = time.monotonic()
        try:
            backend_run = backend.run(context, events)
            result = backend.normalize(context, backend_run)
            _atomic_json(job_dir / "job-result.json", result.to_dict())
            state.update_job(
                job.id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                result=str((job_dir / "job-result.json").relative_to(jobs_dir.parent)),
            )
            events.emit(
                RunEvent(
                    kind="job_end",
                    run_id=run_id,
                    job_id=job.id,
                    phase="reporting",
                    completed=result.counts.completed,
                    total=plan.total,
                    unit=plan.unit,
                    message="completed",
                )
            )
            return result
        except InterruptedError:
            result = self._failed_result(
                job,
                "cancelled",
                "run cancelled",
                plan.total,
                time.monotonic() - started,
            )
            _atomic_json(job_dir / "job-result.json", result.to_dict())
            state.update_job(
                job.id,
                status="cancelled",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            events.emit(
                RunEvent(
                    kind="job_cancelled",
                    run_id=run_id,
                    job_id=job.id,
                    message="cancelled",
                    level="warning",
                )
            )
            return result
        except Exception as exc:
            result = self._failed_result(
                job, "failed", str(exc), plan.total, time.monotonic() - started
            )
            _atomic_json(job_dir / "job-result.json", result.to_dict())
            state.update_job(
                job.id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            events.emit(
                RunEvent(
                    kind="job_failed",
                    run_id=run_id,
                    job_id=job.id,
                    message=str(exc),
                    level="error",
                )
            )
            return result
        finally:
            reservations.release(job.target, granted)

    def _failed_result(
        self,
        job: JobConfig,
        status: str,
        error: str,
        planned: int | None = None,
        duration: float = 0.0,
    ) -> DatasetResult:
        return DatasetResult(
            job_id=job.id,
            backend=job.backend,
            dataset=job.dataset,
            status=status,
            primary_metric=None,
            metrics={},
            counts=ResultCounts(planned=planned, failed=1),
            duration_seconds=duration,
            error=error,
        )

    def _make_manifest(self, run_id: str, fingerprint: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "suite": self.suite_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "framework_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "repository": _git_identity(),
            "packages": {
                name: _package_version(name)
                for name in ("evalscope", "bfcl-eval", "soundfile", "PyYAML", "rich")
            },
            "sample_retention": self.config.runtime.sample_retention,
            "targets": {
                name: {
                    "protocol": target.protocol,
                    "base_url": target.base_url,
                    "model": target.model,
                    "api_key_env": target.api_key_env,
                    "max_concurrency": target.max_concurrency,
                    "provenance": target.provenance,
                }
                for name, target in self.config.targets.items()
            },
        }


def load_resume_config(run_dir: Path) -> tuple[AppConfig, str]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return load_config(run_dir / "effective-config.yaml"), manifest["suite"]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_identity() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _configured_backend_secrets(jobs: tuple[JobConfig, ...]) -> list[str]:
    values: list[str] = []
    for job in jobs:
        for key, env_name in job.backend_args.items():
            if key.endswith("_env") and isinstance(env_name, str):
                value = os.environ.get(env_name)
                if value:
                    values.append(value)
    return values
