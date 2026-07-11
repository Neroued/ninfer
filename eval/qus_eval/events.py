from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

EVENT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunEvent:
    kind: str
    run_id: str
    job_id: str | None = None
    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    message: str = ""
    level: str = "info"
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)
    schema_version: int = EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventRenderer(Protocol):
    def handle(self, event: RunEvent) -> None: ...
    def close(self) -> None: ...


class NullRenderer:
    def handle(self, event: RunEvent) -> None:
        del event

    def close(self) -> None:
        pass


class EventSink:
    def __init__(
        self,
        path: Path,
        logger: logging.Logger,
        renderer: EventRenderer | None = None,
        secrets: list[str] | None = None,
    ):
        self._path = path
        self._logger = logger
        self._renderer = renderer or NullRenderer()
        self._secrets = [value for value in (secrets or []) if value]
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def emit(self, event: RunEvent) -> None:
        safe = self._redact_event(event)
        with self._lock:
            self._file.write(json.dumps(safe.to_dict(), ensure_ascii=False) + "\n")
            self._file.flush()
            log_method = getattr(self._logger, safe.level, self._logger.info)
            prefix = f"[{safe.job_id}] " if safe.job_id else ""
            progress = ""
            if safe.completed is not None:
                progress = f" {safe.completed}/{safe.total if safe.total is not None else '?'} {safe.unit or ''}"
            log_method("%s%s%s", prefix, safe.message or safe.kind, progress)
            self._renderer.handle(safe)

    def close(self) -> None:
        with self._lock:
            self._renderer.close()
            self._file.close()

    def _redact_event(self, event: RunEvent) -> RunEvent:
        data = event.to_dict()
        encoded = json.dumps(data, ensure_ascii=False)
        for secret in self._secrets:
            encoded = encoded.replace(secret, "<redacted>")
        return RunEvent(**json.loads(encoded))
