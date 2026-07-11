from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import JobConfig
from ..events import EventSink
from ..result import DatasetResult
from ..secrets import ResolvedTarget


class BackendDependencyError(RuntimeError):
    """Raised when a configured backend dependency is absent or incompatible."""


@dataclass(frozen=True)
class WorkPlan:
    total: int | None
    unit: str = "samples"
    subsets: tuple[str, ...] = ()
    supports_resume: bool = False
    requirements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class RunContext:
    run_id: str
    job: JobConfig
    job_dir: Path
    target: ResolvedTarget | None
    granted_concurrency: int
    cancel_event: threading.Event
    resume: bool
    plan: WorkPlan
    sample_retention: str = "all"


@dataclass
class BackendRun:
    started_monotonic: float
    finished_monotonic: float
    raw_result: Any
    artifacts: list[Path] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_monotonic - self.started_monotonic)


class EvaluationBackend(Protocol):
    name: str

    def validate(
        self, job: JobConfig, target: ResolvedTarget | None, for_run: bool = False
    ) -> None: ...
    def plan(self, job: JobConfig, target: ResolvedTarget | None) -> WorkPlan: ...
    def run(self, context: RunContext, events: EventSink) -> BackendRun: ...
    def normalize(self, context: RunContext, run: BackendRun) -> DatasetResult: ...
