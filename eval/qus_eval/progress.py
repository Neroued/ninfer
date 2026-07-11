from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

from .events import RunEvent


@dataclass
class _PlainState:
    last_print: float = 0.0
    started: float = 0.0


class ProgressRenderer:
    """Render structured progress with Rich on a TTY and heartbeats otherwise."""

    def __init__(
        self,
        enabled: bool,
        heartbeat_seconds: float,
        refresh_seconds: float = 1.0,
        stream=None,
    ):
        self.enabled = enabled
        self.heartbeat_seconds = heartbeat_seconds
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._plain: dict[str, _PlainState] = {}
        self._rich = None
        self._tasks: dict[str, int] = {}
        if enabled and getattr(self.stream, "isatty", lambda: False)():
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._rich = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=Console(file=self.stream),
                refresh_per_second=max(0.25, min(10.0, 1.0 / refresh_seconds)),
            )
            self._rich.start()

    def handle(self, event: RunEvent) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._rich is not None:
                self._handle_rich(event)
            else:
                self._handle_plain(event)

    def _handle_rich(self, event: RunEvent) -> None:
        if event.job_id is None:
            job = "__overall__"
            description = "overall"
        else:
            job = event.job_id
            description = f"{job} [{event.phase or event.kind}]"
        task_id = self._tasks.get(job)
        if task_id is None:
            total = event.total if event.total is not None else None
            task_id = self._rich.add_task(job, total=total)
            self._tasks[job] = task_id
        kwargs = {"description": description}
        if event.total is not None:
            kwargs["total"] = event.total
        if event.completed is not None:
            kwargs["completed"] = event.completed
        if event.kind in {"job_end", "job_failed", "job_cancelled", "job_skipped"}:
            task = self._rich.tasks[task_id]
            if task.total is None:
                kwargs["total"] = max(task.completed, 1)
                kwargs["completed"] = max(task.completed, 1)
        self._rich.update(task_id, **kwargs)
        if event.job_id is not None and event.kind in {
            "job_end",
            "job_failed",
            "job_cancelled",
            "job_skipped",
        }:
            overall = self._tasks.get("__overall__")
            if overall is not None:
                self._rich.advance(overall, 1)

    def _handle_plain(self, event: RunEvent) -> None:
        now = time.monotonic()
        key = event.job_id or "__overall__"
        state = self._plain.setdefault(key, _PlainState())
        if state.started == 0.0:
            state.started = now
        important = event.kind in {
            "run_start",
            "run_end",
            "job_start",
            "job_end",
            "job_failed",
            "job_cancelled",
            "job_skipped",
            "retry",
        }
        if not important and now - state.last_print < self.heartbeat_seconds:
            return
        state.last_print = now
        count = ""
        if event.completed is not None:
            count = f" {event.completed}/{event.total if event.total is not None else '?'} {event.unit or ''}"
        elapsed = max(0.0, now - state.started)
        timing = f" elapsed={elapsed:.1f}s"
        if event.completed and elapsed > 0:
            rate = event.completed / elapsed
            timing += f" rate={rate:.2f}/{event.unit or 'unit'}/s"
            if event.total is not None and event.total >= event.completed:
                timing += f" eta={(event.total - event.completed) / rate:.1f}s"
        print(
            f"[{event.timestamp}] [{event.job_id or 'overall'}] {event.phase or event.kind}{count}{timing} {event.message}".rstrip(),
            file=self.stream,
            flush=True,
        )

    def close(self) -> None:
        with self._lock:
            if self._rich is not None:
                self._rich.stop()
                self._rich = None
