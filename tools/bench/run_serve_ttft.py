#!/usr/bin/env python3
"""Run one audited TTFT request graph against an already-running ninfer-serve."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.ttft.cases import CASES, get_case, run_case
from tools.bench.ttft.corpus import DEFAULT_MANIFEST, Corpus
from tools.bench.ttft.execution import CaseContext
from tools.ninfer_serve.client import NInferServeClient


_PROGRESS_STOP = object()


class StderrProgress:
    """Serialize concurrent request milestones without putting stderr I/O on request workers."""

    def __init__(self, heartbeat_seconds: float = 5.0) -> None:
        self._started_ns = time.perf_counter_ns()
        self._heartbeat_seconds = heartbeat_seconds
        self._events: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._consume,
            name="ttft-progress",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _line(self, stage: str, occurred_ns: int, fields: dict[str, Any]) -> None:
        elapsed_seconds = max(0.0, (occurred_ns - self._started_ns) / 1e9)
        details = " ".join(
            f"{name}={self._value(value)}"
            for name, value in fields.items()
            if value is not None
        )
        suffix = f" {details}" if details else ""
        try:
            print(
                f"[ttft +{elapsed_seconds:9.3f}s] {stage}{suffix}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            # Losing a progress stream must not invalidate a completed measurement.
            pass

    def _consume(self) -> None:
        active: dict[str, tuple[str, int]] = {}
        last_stage = "startup"
        last_event_ns = self._started_ns
        while True:
            try:
                item = self._events.get(timeout=self._heartbeat_seconds)
            except queue.Empty:
                now = time.perf_counter_ns()
                active_text = [
                    f"{key}:{stage}"
                    for key, (stage, _) in sorted(active.items())
                ]
                self._line(
                    "heartbeat",
                    now,
                    {
                        "last_stage": last_stage,
                        "quiet_s": (now - last_event_ns) / 1e9,
                        "active": active_text,
                    },
                )
                continue

            if item is _PROGRESS_STOP:
                return
            stage, occurred_ns, fields = item
            assert isinstance(stage, str)
            assert isinstance(occurred_ns, int)
            assert isinstance(fields, dict)
            self._line(stage, occurred_ns, fields)
            last_stage = stage
            last_event_ns = occurred_ns

            role = fields.get("role")
            order = fields.get("order")
            if isinstance(role, str) and isinstance(order, int):
                key = f"{order}:{role}"
                if stage == "request.started":
                    active[key] = (stage, occurred_ns)
                elif key in active:
                    active[key] = (stage, occurred_ns)
                if stage == "request.finished":
                    active.pop(key, None)

    def emit(self, stage: str, occurred_ns: int, fields: dict[str, Any]) -> None:
        if not self._closed:
            self._events.put((stage, occurred_ns, fields))

    def event(self, stage: str, **fields: Any) -> None:
        self.emit(stage, time.perf_counter_ns(), fields)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.put(_PROGRESS_STOP)
        self._thread.join()


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key-env", default="NINFER_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    definition = get_case(args.case)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    client = NInferServeClient(args.base_url.rstrip("/"), args.timeout_seconds, api_key)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    progress = StderrProgress()
    progress.event(
        "case.selected",
        case=definition.name,
        category=definition.category,
        protocol=definition.protocol,
        profile=args.profile_label,
        expected_profile=definition.profile,
        timeout_s=args.timeout_seconds,
    )

    if args.profile_label == definition.profile:
        try:
            progress.event("corpus.load", manifest=str(DEFAULT_MANIFEST))
            corpus = Corpus()
            progress.event(
                "corpus.ready",
                shapes=len(corpus.manifest["shapes"]),
                shared=len(corpus.manifest["shared"]),
                media=len(corpus.manifest["media"]),
            )
            progress.event("model.discover", base_url=args.base_url.rstrip("/"))
            model = client.discover_model()
            progress.event("model.ready", model=model)
            context = CaseContext(client, model, args.timeout_seconds, progress.emit)
            progress.event("case.graph_start", description=definition.description)
            result = run_case(definition, context, corpus, args.profile_label)
            result["server"] = {"base_url": args.base_url.rstrip("/"), "model": model}
        except Exception as error:
            progress.event(
                "case.setup_failed",
                error=f"{type(error).__name__}: {error}",
            )
            result = {
                "artifact_type": "ninfer_serve_ttft_run",
                "schema_version": 1,
                "case": definition.name,
                "protocol": definition.protocol,
                "category": definition.category,
                "description": definition.description,
                "profile_label": args.profile_label,
                "expected_profile_label": definition.profile,
                "corpus_ids": list(definition.corpus_ids),
                "status": "not_constructed",
                "constructed": False,
                "failed_conditions": [
                    {
                        "expression": "benchmark setup completed",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                ],
                "server": {"base_url": args.base_url.rstrip("/"), "model": None},
                "requests": [],
            }
    else:
        # Label validation is deliberately possible without contacting or interpreting Serve.
        progress.event(
            "case.profile_mismatch",
            profile=args.profile_label,
            expected_profile=definition.profile,
        )
        context = CaseContext(client, "", args.timeout_seconds, progress.emit)
        result = run_case(definition, context, None, args.profile_label)
        result["server"] = {"base_url": args.base_url.rstrip("/"), "model": None}

    result["started_at"] = started
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        progress.event("result.write", path=str(args.output))
        try:
            _write(args.output, result)
        except Exception as error:
            progress.event(
                "result.write_failed",
                path=str(args.output),
                error=f"{type(error).__name__}: {error}",
            )
            raise
        progress.event(
            "case.finished",
            status=result.get("status"),
            constructed=result.get("constructed"),
            requests=len(result.get("requests", [])),
            output=str(args.output),
        )
    finally:
        progress.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("constructed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
