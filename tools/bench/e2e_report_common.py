#!/usr/bin/env python3
"""Shared helpers for M2.8 e2e report validation, comparison, and summaries."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


class ReportValidationError(ValueError):
    """Raised when an e2e report does not satisfy the M2.8 schema subset."""


TOP_LEVEL_FIELDS = (
    "schema_version",
    "artifact_type",
    "status",
    "run",
    "environment",
    "engine",
    "weights",
    "memory",
    "summary",
    "cases",
)

CASE_FIELDS = (
    "name",
    "fixture_set",
    "fixture_manifest_path",
    "fixture_manifest_sha256",
    "prompt_ids_path",
    "prompt_ids_sha256",
    "prompt_tokens",
    "requested_max_new_tokens",
    "eos_token_id",
    "max_context",
    "decode_loop_tokens_requested",
    "required_max_context",
    "warmup_repeats",
    "measured_repeats",
    "repeats",
    "summary",
)

REPEAT_FIELDS = (
    "repeat_index",
    "prefill_time_s",
    "decode_time_s",
    "e2e_excluding_load_time_s",
    "prompt_tokens",
    "prefill_output_tokens",
    "decode_loop_tokens",
    "generated_tokens_total",
    "decode_eager_tok_s",
    "decode_eager_tok_s_valid",
    "e2e_excluding_load_tok_s",
    "stop_reason",
    "generated_token_ids",
    "memory",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ReportValidationError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportValidationError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportValidationError(f"{label} must be a list")
    return value


def require_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReportValidationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportValidationError(f"{label} must be a finite number")
    return number


def require_int(value: Any, label: str, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReportValidationError(f"{label} must be an integer")
    if min_value is not None and value < min_value:
        raise ReportValidationError(f"{label} must be >= {min_value}")
    return value


def _require_fields(obj: dict[str, Any], fields: Iterable[str], label: str) -> None:
    for field in fields:
        if field not in obj:
            raise ReportValidationError(f"{label} missing required field: {field}")


def _validate_repeat(repeat: dict[str, Any], case_name: str) -> None:
    _require_fields(repeat, REPEAT_FIELDS, f"repeat in {case_name}")
    ids = require_list(repeat["generated_token_ids"], f"{case_name}.generated_token_ids")
    if not ids or not all(isinstance(value, int) and value >= 0 for value in ids):
        raise ReportValidationError(f"{case_name}.generated_token_ids must be nonempty nonnegative ints")

    decode_loop_tokens = require_int(repeat["decode_loop_tokens"], f"{case_name}.decode_loop_tokens", 0)
    generated_tokens_total = require_int(
        repeat["generated_tokens_total"],
        f"{case_name}.generated_tokens_total",
        1,
    )
    if generated_tokens_total != len(ids):
        raise ReportValidationError(f"{case_name}.generated_tokens_total must match generated_token_ids")
    if generated_tokens_total != 1 + decode_loop_tokens:
        raise ReportValidationError(f"{case_name}.generated_tokens_total must equal 1 + decode_loop_tokens")

    valid = repeat["decode_eager_tok_s_valid"]
    if decode_loop_tokens == 0:
        if repeat["decode_eager_tok_s"] is not None or valid is not False:
            raise ReportValidationError(f"{case_name} zero decode repeat must use null/false throughput")
    else:
        if valid is not True:
            raise ReportValidationError(f"{case_name} nonzero decode repeat must mark throughput valid")
        require_number(repeat["decode_eager_tok_s"], f"{case_name}.decode_eager_tok_s")

    require_mapping(repeat["memory"], f"{case_name}.repeat.memory")


def _validate_case(case: dict[str, Any]) -> None:
    _require_fields(case, CASE_FIELDS, "case")
    name = str(case["name"])
    if not name:
        raise ReportValidationError("case name must be nonempty")

    prompt_tokens = require_int(case["prompt_tokens"], f"{name}.prompt_tokens", 1)
    requested = require_int(case["requested_max_new_tokens"], f"{name}.requested_max_new_tokens", 1)
    require_int(case["eos_token_id"], f"{name}.eos_token_id", -1)
    decode_requested = require_int(
        case["decode_loop_tokens_requested"],
        f"{name}.decode_loop_tokens_requested",
        0,
    )
    required_context = require_int(case["required_max_context"], f"{name}.required_max_context", 1)
    if decode_requested != max(requested - 1, 0):
        raise ReportValidationError(f"{name}.decode_loop_tokens_requested formula mismatch")
    if required_context != prompt_tokens + decode_requested:
        raise ReportValidationError(f"{name}.required_max_context formula mismatch")

    require_int(case["warmup_repeats"], f"{name}.warmup_repeats", 0)
    measured_repeats = require_int(case["measured_repeats"], f"{name}.measured_repeats", 1)
    repeats = require_list(case["repeats"], f"{name}.repeats")
    if len(repeats) != measured_repeats:
        raise ReportValidationError(f"{name}.repeats length must equal measured_repeats")
    summary = require_mapping(case["summary"], f"{name}.summary")
    checked_repeats = []
    for repeat in repeats:
        checked_repeat = require_mapping(repeat, f"{name}.repeat")
        _validate_repeat(checked_repeat, name)
        checked_repeats.append(checked_repeat)

    if "decode_eager_tok_s_median" not in summary:
        raise ReportValidationError(f"{name}.summary missing decode_eager_tok_s_median")
    has_valid_decode_throughput = any(
        repeat["decode_eager_tok_s_valid"] is True for repeat in checked_repeats
    )
    decode_median = summary["decode_eager_tok_s_median"]
    if has_valid_decode_throughput:
        require_number(decode_median, f"{name}.summary.decode_eager_tok_s_median")
    elif decode_median is not None:
        raise ReportValidationError(
            f"{name}.summary zero decode throughput median must be null"
        )


def validate_report(report: dict[str, Any], require_ok: bool = True) -> None:
    _require_fields(report, TOP_LEVEL_FIELDS, "report")
    if report["schema_version"] != 1:
        raise ReportValidationError("unsupported schema_version")
    if report["artifact_type"] != "qus_e2e_benchmark_report":
        raise ReportValidationError("artifact_type must be qus_e2e_benchmark_report")
    if require_ok and report["status"] != "ok":
        raise ReportValidationError("report status must be ok")

    require_mapping(report["run"], "run")
    require_mapping(report["environment"], "environment")
    require_mapping(report["engine"], "engine")
    require_mapping(report["weights"], "weights")
    require_mapping(report["memory"], "memory")
    require_mapping(report["summary"], "summary")
    cases = require_list(report["cases"], "cases")
    if not cases:
        raise ReportValidationError("cases must be nonempty")
    for case in cases:
        _validate_case(require_mapping(case, "case"))


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for case in require_list(report.get("cases"), "cases"):
        obj = require_mapping(case, "case")
        name = str(obj.get("name", ""))
        if name in out:
            raise ReportValidationError(f"duplicate case name: {name}")
        out[name] = obj
    return out


def generated_ids_by_repeat(case: dict[str, Any]) -> list[list[int]]:
    return [list(repeat["generated_token_ids"]) for repeat in require_list(case["repeats"], "repeats")]


def nested_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def pct_change(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0 if new == 0.0 else math.inf
    return (new - old) / old


def numeric_summary(case: dict[str, Any], key: str) -> float | None:
    value = require_mapping(case.get("summary"), "case.summary").get(key)
    if value is None:
        return None
    return require_number(value, f"case.summary.{key}")
