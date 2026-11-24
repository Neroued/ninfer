from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def prediction_failed(record: dict[str, Any]) -> bool:
    output = record.get("model_output") or {}
    if output.get("error") is not None:
        return True
    choices = output.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if isinstance(content, str):
        try:
            wrapped = json.loads(content)
        except json.JSONDecodeError:
            return False
        # BFCL serializes provider failures as assistant content.
        return (
            isinstance(wrapped, dict) and {"error", "error_message"} <= wrapped.keys()
        )
    return False


def summarize_records(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = {
        "predictions": 0,
        "failed": 0,
        "truncated": 0,
        "empty_answers": 0,
        "unparsed_answers": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for path in sorted((job_dir / "predictions").rglob("*.jsonl")):
        for record in records(path):
            diagnostics["predictions"] += 1
            if prediction_failed(record):
                diagnostics["failed"] += 1
                continue
            output = record.get("model_output") or {}
            choices = output.get("choices") or []
            diagnostics["truncated"] += any(
                choice.get("stop_reason") in {"max_tokens", "length"}
                for choice in choices
            )
            content = (
                (choices[0].get("message") or {}).get("content") if choices else None
            )
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
            diagnostics["empty_answers"] += not content or not str(content).strip()
            usage = output.get("usage") or {}
            for key in ("input_tokens", "output_tokens"):
                diagnostics[key] += int(usage.get(key) or 0)

    questions = []
    for path in sorted((job_dir / "reviews").rglob("*.jsonl")):
        groups: dict[str, dict[str, Any]] = {}
        for record in records(path):
            sample = record.get("sample_score") or {}
            score = sample.get("score") or {}
            if "extracted_prediction" in score:
                extracted = score["extracted_prediction"]
                diagnostics["unparsed_answers"] += (
                    extracted is None or not str(extracted).strip()
                )
            accuracy = (score.get("value") or {}).get("acc")
            if not isinstance(accuracy, (int, float)):
                continue
            group_id = sample.get("group_id")
            if group_id is None:
                group_id = sample.get("sample_id", record.get("index"))
            group = groups.setdefault(
                str(group_id),
                {
                    "subset": path.stem,
                    "question_id": group_id,
                    "samples": 0,
                    "correct": 0,
                },
            )
            group["samples"] += 1
            group["correct"] += int(accuracy == 1)
        questions.extend(groups.values())
    if questions:
        diagnostics["questions"] = len(questions)
        diagnostics["correct"] = sum(question["correct"] for question in questions)
    return diagnostics, questions


def prepare_resume(job_dir: Path) -> None:
    """Archive failed requests and their reviews; retain successful inference."""
    for path in sorted((job_dir / "predictions").rglob("*.jsonl")):
        failed = {r["index"] for r in records(path) if prediction_failed(r)}
        if not failed:
            continue
        relative = path.relative_to(job_dir / "predictions")
        for kind in ("predictions", "reviews"):
            source_path = job_dir / kind / relative
            if not source_path.exists():
                continue
            archive = job_dir / "failed-attempts" / kind / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            tmp = source_path.with_suffix(".jsonl.tmp")
            with (
                source_path.open(encoding="utf-8") as source,
                tmp.open("w", encoding="utf-8") as kept,
                archive.open("a", encoding="utf-8") as removed,
            ):
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    destination = removed if record.get("index") in failed else kept
                    destination.write(line)
            tmp.replace(source_path)
