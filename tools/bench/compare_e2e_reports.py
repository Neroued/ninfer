#!/usr/bin/env python3
"""Compare two M2.8 e2e benchmark reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench import e2e_report_common as common  # noqa: E402

PERF_THRESHOLD = 0.05
MEMORY_THRESHOLD = 0.05
MEMORY_ABS_THRESHOLD = 64 * 1024 * 1024


@dataclass
class CompareResult:
    failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def add_failure(self, code: str, message: str, **details: Any) -> None:
        self.failures.append({"code": code, "message": message, **details})

    def add_warning(self, code: str, message: str, **details: Any) -> None:
        self.warnings.append({"code": code, "message": message, **details})

    def status(self) -> str:
        if self.failures:
            return "fail"
        if self.warnings:
            return "warning"
        return "ok"

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_type": "qus_e2e_compare_result",
            "schema_version": 1,
            "status": self.status(),
            "failures": self.failures,
            "warnings": self.warnings,
        }


def _promote(result: CompareResult, fail: bool, code: str, message: str, **details: Any) -> None:
    if fail:
        result.add_failure(code, message, **details)
    else:
        result.add_warning(code, message, **details)


def _compare_top_level(baseline: dict[str, Any], candidate: dict[str, Any], result: CompareResult) -> None:
    old_workspace = common.nested_get(baseline, "engine.workspace_lifetime_policy")
    new_workspace = common.nested_get(candidate, "engine.workspace_lifetime_policy")
    if old_workspace != new_workspace:
        result.add_warning(
            "workspace_lifetime_policy_changed",
            "workspace lifetime policy changed",
            baseline=old_workspace,
            candidate=new_workspace,
        )

    old_load = common.nested_get(baseline, "weights.load_strategy")
    new_load = common.nested_get(candidate, "weights.load_strategy")
    if old_load != new_load:
        result.add_warning(
            "load_strategy_changed",
            "load strategy changed",
            baseline=old_load,
            candidate=new_load,
        )

    old_scope = common.nested_get(baseline, "memory.accounting_scope")
    new_scope = common.nested_get(candidate, "memory.accounting_scope")
    if old_scope != new_scope:
        result.add_warning(
            "memory_accounting_scope_changed",
            "memory accounting scope changed",
            baseline=old_scope,
            candidate=new_scope,
        )


def _compare_q5090_identity(baseline: dict[str, Any], candidate: dict[str, Any], result: CompareResult) -> None:
    fields = ("q5090_path", "q5090_file_size_bytes", "q5090_sha256")
    missing = {
        "baseline": [field for field in fields if field not in baseline["weights"]],
        "candidate": [field for field in fields if field not in candidate["weights"]],
    }
    if missing["baseline"] or missing["candidate"]:
        result.add_failure("q5090_identity_missing", "q5090 identity fields are missing", fields=missing)
        return

    changed = {
        field: (baseline["weights"][field], candidate["weights"][field])
        for field in fields
        if baseline["weights"][field] != candidate["weights"][field]
    }
    if changed:
        result.add_failure("q5090_identity_changed", "q5090 identity changed", fields=changed)


def _compare_case_identity(base_case: dict[str, Any], cand_case: dict[str, Any], result: CompareResult) -> None:
    fields = (
        "fixture_set",
        "fixture_manifest_path",
        "fixture_manifest_sha256",
        "prompt_format",
        "messages_path",
        "messages_sha256",
        "rendered_prompt_sha256",
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template_kwargs",
        "stop_token_ids",
        "prompt_ids_path",
        "prompt_ids_sha256",
        "prompt_tokens",
        "requested_max_new_tokens",
    )
    changed = {
        field: (base_case.get(field), cand_case.get(field))
        for field in fields
        if not _identity_value_equal(base_case.get(field), cand_case.get(field))
    }
    if changed:
        result.add_failure(
            "case_identity_changed",
            f"case identity changed for {base_case['name']}",
            case=base_case["name"],
            fields=changed,
        )


def _identity_value_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if set(left) != set(right):
            return False
        return all(_identity_value_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        if len(left) != len(right):
            return False
        return all(_identity_value_equal(old, new) for old, new in zip(left, right))
    return left == right


def _compare_tokens(base_case: dict[str, Any], cand_case: dict[str, Any], result: CompareResult) -> None:
    base_ids = common.generated_ids_by_repeat(base_case)
    cand_ids = common.generated_ids_by_repeat(cand_case)
    if len(base_ids) != len(cand_ids):
        result.add_failure(
            "repeat_count_changed",
            f"repeat count changed for {base_case['name']}",
            case=base_case["name"],
            baseline=len(base_ids),
            candidate=len(cand_ids),
        )
        return
    for index, (old, new) in enumerate(zip(base_ids, cand_ids)):
        if old != new:
            result.add_failure(
                "generated_token_ids_changed",
                f"generated token ids changed for {base_case['name']} repeat {index}",
                case=base_case["name"],
                repeat_index=index,
            )


def _compare_perf(
    base_case: dict[str, Any],
    cand_case: dict[str, Any],
    result: CompareResult,
    fail_on_performance_regression: bool,
) -> None:
    for key in (
        "prefill_prompt_tok_s_median",
        "decode_eager_tok_s_median",
        "e2e_excluding_load_tok_s_median",
    ):
        old = common.numeric_summary(base_case, key)
        new = common.numeric_summary(cand_case, key)
        if old is not None and new is not None and common.pct_change(old, new) < -PERF_THRESHOLD:
            _promote(
                result,
                fail_on_performance_regression,
                "throughput_drop",
                f"{key} dropped for {base_case['name']}",
                case=base_case["name"],
                metric=key,
                baseline=old,
                candidate=new,
            )

    for key in ("prefill_time_s_median", "decode_time_s_median"):
        old = common.numeric_summary(base_case, key)
        new = common.numeric_summary(cand_case, key)
        if old is not None and new is not None and common.pct_change(old, new) > PERF_THRESHOLD:
            _promote(
                result,
                fail_on_performance_regression,
                "phase_time_increase",
                f"{key} increased for {base_case['name']}",
                case=base_case["name"],
                metric=key,
                baseline=old,
                candidate=new,
            )


def _compare_memory(
    base_case: dict[str, Any],
    cand_case: dict[str, Any],
    result: CompareResult,
    fail_on_memory_regression: bool,
) -> None:
    for key in (
        "max_weight_arena_peak_used_bytes",
        "max_cache_arena_peak_used_bytes",
        "max_workspace_arena_peak_used_bytes",
    ):
        old = common.numeric_summary(base_case, key)
        new = common.numeric_summary(cand_case, key)
        if old is None or new is None:
            continue
        increase = new - old
        if increase > MEMORY_ABS_THRESHOLD and common.pct_change(old, new) > MEMORY_THRESHOLD:
            _promote(
                result,
                fail_on_memory_regression,
                "memory_peak_increase",
                f"{key} increased for {base_case['name']}",
                case=base_case["name"],
                metric=key,
                baseline=old,
                candidate=new,
            )


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    required_cases: list[str] | None = None,
    fail_on_performance_regression: bool = False,
    fail_on_memory_regression: bool = False,
    skip_token_id_check: bool = False,
) -> CompareResult:
    result = CompareResult()
    try:
        common.validate_report(baseline)
        common.validate_report(candidate)
    except common.ReportValidationError as exc:
        result.add_failure("schema_invalid", str(exc))
        return result

    _compare_top_level(baseline, candidate, result)
    if not skip_token_id_check:
        _compare_q5090_identity(baseline, candidate, result)

    try:
        base_cases = common.case_map(baseline)
        cand_cases = common.case_map(candidate)
    except common.ReportValidationError as exc:
        result.add_failure("schema_invalid", str(exc))
        return result

    expected_cases = required_cases or list(base_cases.keys())
    for name in expected_cases:
        if name not in base_cases:
            result.add_failure("missing_baseline_case", f"baseline missing case {name}", case=name)
            continue
        if name not in cand_cases:
            result.add_failure("missing_candidate_case", f"candidate missing case {name}", case=name)
            continue

        base_case = base_cases[name]
        cand_case = cand_cases[name]
        _compare_case_identity(base_case, cand_case, result)
        if not skip_token_id_check:
            _compare_tokens(base_case, cand_case, result)
        _compare_perf(base_case, cand_case, result, fail_on_performance_regression)
        _compare_memory(base_case, cand_case, result, fail_on_memory_regression)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--required-case", action="append", default=[])
    parser.add_argument("--fail-on-performance-regression", action="store_true")
    parser.add_argument("--fail-on-memory-regression", action="store_true")
    parser.add_argument("--skip-token-id-check", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = compare_reports(
            common.load_json(args.baseline),
            common.load_json(args.candidate),
            required_cases=args.required_case or None,
            fail_on_performance_regression=args.fail_on_performance_regression,
            fail_on_memory_regression=args.fail_on_memory_regression,
            skip_token_id_check=args.skip_token_id_check,
        )
    except Exception as exc:
        result = CompareResult()
        result.add_failure("schema_invalid", str(exc))

    text = json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
