from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1


@dataclass
class ResultCounts:
    planned: int | None = None
    completed: int = 0
    scored: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class DatasetResult:
    job_id: str
    backend: str
    dataset: str
    status: str
    primary_metric: str | None
    metrics: dict[str, Any]
    counts: ResultCounts
    duration_seconds: float
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetResult":
        value = dict(data)
        value["counts"] = ResultCounts(**value["counts"])
        return cls(**value)


def write_summary(
    run_dir: Path, run_id: str, status: str, results: list[DatasetResult]
) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "results": [result.to_dict() for result in results],
    }
    _atomic_json(run_dir / "summary.json", payload)
    lines = [f"# Evaluation Summary: {run_id}", "", f"Status: `{status}`", ""]
    lines.extend(
        [
            "| Job | Backend | Dataset | Status | Primary metric | Score | Completed | Failed | Duration |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        score = (
            result.metrics.get(result.primary_metric) if result.primary_metric else None
        )
        score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "-"
        lines.append(
            f"| {result.job_id} | {result.backend} | {result.dataset} | {result.status} | "
            f"{result.primary_metric or '-'} | {score_text} | {result.counts.completed} | "
            f"{result.counts.failed} | {result.duration_seconds:.1f}s |"
        )
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
