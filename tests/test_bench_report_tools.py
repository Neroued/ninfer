#!/usr/bin/env python3
"""Unit tests for M2.8 e2e report comparison and summary tools."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bench import e2e_report_common as common  # noqa: E402
from tools.bench import tokenizer_common  # noqa: E402


SHA_FIXTURE = "1" * 64
SHA_CN_IDS = "2" * 64
SHA_LONG_IDS = "3" * 64
SHA_CN_MESSAGES = "4" * 64
SHA_LONG_MESSAGES = "5" * 64
SHA_CN_RENDERED = "6" * 64
SHA_LONG_RENDERED = "7" * 64


def arena(name: str, peak: int) -> dict[str, object]:
    return {
        "name": name,
        "present": True,
        "capacity_bytes": peak * 2,
        "used_bytes": peak,
        "peak_used_bytes": peak,
    }


def repeat(
    tokens: list[int],
    repeat_index: int,
    prompt_tokens: int,
    stop_reason: str,
    tok_s: float = 100.0,
    workspace_peak: int = 256 * 1024 * 1024,
) -> dict[str, object]:
    decode_loop_tokens = max(len(tokens) - 1, 0)
    decode_time_s = 0.01 if decode_loop_tokens else 0.0
    return {
        "repeat_index": repeat_index,
        "prefill_time_s": 0.02,
        "decode_time_s": decode_time_s,
        "e2e_excluding_load_time_s": 0.02 + decode_time_s,
        "prompt_tokens": prompt_tokens,
        "prefill_output_tokens": 1,
        "decode_loop_tokens": decode_loop_tokens,
        "generated_tokens_total": len(tokens),
        "decode_eager_tok_s": tok_s if decode_loop_tokens else None,
        "decode_eager_tok_s_valid": bool(decode_loop_tokens),
        "e2e_excluding_load_tok_s": 80.0,
        "stop_reason": stop_reason,
        "generated_token_ids": tokens,
        "memory": {"arenas": [arena("workspace", workspace_peak)]},
    }


def case(
    name: str,
    tokens: list[int],
    tok_s: float = 100.0,
    workspace_peak: int = 256 * 1024 * 1024,
) -> dict[str, object]:
    prompt_tokens = 2048 if name == "long_2k" else 4
    requested_max_new_tokens = 128 if name == "cn_short" else 1
    measured_repeats = 3 if name == "cn_short" else 1
    decode_loop_tokens_requested = max(requested_max_new_tokens - 1, 0)
    required_max_context = prompt_tokens + decode_loop_tokens_requested
    stop_reason = "max_new_tokens" if len(tokens) == requested_max_new_tokens else "eos_token"
    return {
        "name": name,
        "fixture_set": "m2.8-v1",
        "fixture_manifest_path": "bench/fixtures/prompts/m2.8-v1.manifest.json",
        "fixture_manifest_sha256": SHA_FIXTURE,
        "prompt_format": tokenizer_common.PROMPT_FORMAT,
        "messages_path": f"bench/fixtures/prompts/{name}.messages.json",
        "messages_sha256": SHA_LONG_MESSAGES if name == "long_2k" else SHA_CN_MESSAGES,
        "rendered_prompt_sha256": SHA_LONG_RENDERED if name == "long_2k" else SHA_CN_RENDERED,
        "add_generation_prompt": tokenizer_common.ADD_GENERATION_PROMPT,
        "add_special_tokens": tokenizer_common.ADD_SPECIAL_TOKENS,
        "chat_template_kwargs": dict(tokenizer_common.CHAT_TEMPLATE_KWARGS),
        "stop_token_ids": list(tokenizer_common.STOP_TOKEN_IDS),
        "prompt_ids_path": f"bench/fixtures/prompts/{name}.ids",
        "prompt_ids_sha256": SHA_LONG_IDS if name == "long_2k" else SHA_CN_IDS,
        "prompt_tokens": prompt_tokens,
        "requested_max_new_tokens": requested_max_new_tokens,
        "max_context": 4096,
        "decode_loop_tokens_requested": decode_loop_tokens_requested,
        "required_max_context": required_max_context,
        "warmup_repeats": 1 if name == "cn_short" else 0,
        "measured_repeats": measured_repeats,
        "repeats": [
            repeat(
                tokens,
                repeat_index=repeat_index,
                prompt_tokens=prompt_tokens,
                stop_reason=stop_reason,
                tok_s=tok_s,
                workspace_peak=workspace_peak,
            )
            for repeat_index in range(measured_repeats)
        ],
        "summary": {
            "prefill_time_s_median": 0.02,
            "decode_time_s_median": 0.01 if len(tokens) > 1 else 0.0,
            "decode_eager_tok_s_median": tok_s if len(tokens) > 1 else None,
            "e2e_excluding_load_tok_s_median": 80.0,
            "deterministic_token_ids": True,
            "max_weight_arena_peak_used_bytes": 1024,
            "max_cache_arena_peak_used_bytes": 2048,
            "max_workspace_arena_peak_used_bytes": workspace_peak,
        },
    }


def report(
    tok_s: float = 100.0,
    workspace_peak: int = 256 * 1024 * 1024,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "qus_e2e_benchmark_report",
        "status": "ok",
        "run": {
            "binary": "qus_e2e_bench",
            "command": "qus_e2e_bench --case cn_short:bench/fixtures/prompts/cn_short.ids:128",
            "git_commit": "abc123",
            "worktree_dirty": False,
            "load_time_s": 1.5,
        },
        "environment": {
            "cuda_runtime_version": "12.4",
            "cuda_driver_version": "12.4",
            "gpu_name": "test gpu",
            "device_id": 0,
        },
        "engine": {
            "max_context": 4096,
            "workspace_lifetime_policy": "step_reset",
            "decode_metric": "decode_eager_tok_s",
            "sampling_location": "device_argmax",
            "token_readback": "per_step_sync_d2h",
            "includes_token_readback": True,
            "timing_boundary": "host_visible_phase_end",
        },
        "weights": {
            "q5090_path": "out/test.qus",
            "q5090_file_size_bytes": 1234,
            "q5090_sha256": "weights-sha",
            "q5090_conv1d_layout": "runtime_native_conv_dim_by_kernel",
            "load_strategy": "full_file_host_vector_then_h2d_payload_upload",
            "default_weight_arena_policy": "q5090_file_size_plus_256MiB",
            "estimated_host_file_buffer_bytes": 1234,
            "selected_modules": {"text_core": True, "mtp": False, "vision": False},
            "q5090_loaded_payload_bytes": 1000,
            "weight_arena_capacity_bytes": 2000,
            "weight_arena_used_bytes": 1500,
            "weight_arena_peak_used_bytes": 1500,
            "weight_arena_slack_bytes": 500,
            "weight_payload_to_arena_used_overhead_bytes": 500,
        },
        "memory": {
            "accounting_scope": "engine_owned_device_arenas_complete",
            "hidden_device_allocations": False,
            "arenas": [
                arena("weights", 1500),
                arena("cache", 2048),
                arena("workspace", workspace_peak),
            ],
            "q5090_loaded_payload_bytes": 1000,
            "q5090_tensor_count": 10,
            "q5090_quant_count": 6,
            "known_exclusions": [],
        },
        "summary": {"case_count": 2, "load_time_s": 1.5},
        "cases": [
            case("cn_short", [10, 11, 12], tok_s=tok_s, workspace_peak=workspace_peak),
            case("long_2k", [20], tok_s=tok_s, workspace_peak=workspace_peak),
        ],
    }


class ReportCommonTests(unittest.TestCase):
    def test_validate_report_accepts_valid_report(self) -> None:
        common.validate_report(report())

    def test_validate_report_rejects_missing_top_level_field(self) -> None:
        bad = report()
        del bad["weights"]
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_validate_report_rejects_bad_zero_decode_throughput(self) -> None:
        bad = report()
        bad["cases"][1]["repeats"][0]["decode_eager_tok_s"] = 0.0
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_validate_report_rejects_bad_zero_decode_summary_throughput(self) -> None:
        bad = report()
        bad["cases"][1]["summary"]["decode_eager_tok_s_median"] = 0.0
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_case_map_and_generated_ids(self) -> None:
        cases = common.case_map(report())
        self.assertEqual(sorted(cases), ["cn_short", "long_2k"])
        self.assertEqual(
            common.generated_ids_by_repeat(cases["cn_short"]),
            [[10, 11, 12], [10, 11, 12], [10, 11, 12]],
        )

    def test_validate_report_rejects_repeat_count_mismatch(self) -> None:
        bad = report()
        bad["cases"][0]["repeats"].pop()
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_validate_report_rejects_bool_generated_token_id(self) -> None:
        bad = report()
        bad["cases"][0]["repeats"][0]["generated_token_ids"] = [10, True, 12]
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_validate_report_rejects_bool_repeat_index(self) -> None:
        bad = report()
        bad["cases"][0]["repeats"][0]["repeat_index"] = True
        with self.assertRaises(common.ReportValidationError):
            common.validate_report(bad)

    def test_validate_report_rejects_any_eos_token_id(self) -> None:
        bad = report()
        bad["cases"][0]["eos_token_id"] = -1
        with self.assertRaisesRegex(
            common.ReportValidationError,
            "report schema uses stop_token_ids; eos_token_id is not allowed and report must be regenerated",
        ):
            common.validate_report(bad)

    def test_validate_report_rejects_bad_chat_identity_fields(self) -> None:
        mutations = {
            "prompt_format": lambda value: value["cases"][0].__setitem__(
                "prompt_format", "raw-text"
            ),
            "messages_sha256": lambda value: value["cases"][0].__setitem__(
                "messages_sha256", "not-a-sha"
            ),
            "rendered_prompt_sha256": lambda value: value["cases"][0].__setitem__(
                "rendered_prompt_sha256", "A" * 64
            ),
            "chat_template_kwargs": lambda value: value["cases"][0].__setitem__(
                "chat_template_kwargs", {"enable_thinking": True}
            ),
            "chat_template_kwargs_int_bool": lambda value: value["cases"][0].__setitem__(
                "chat_template_kwargs", {"enable_thinking": 0}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                bad = report()
                mutate(bad)
                with self.assertRaises(common.ReportValidationError):
                    common.validate_report(bad)


class CompareReportTests(unittest.TestCase):
    def test_identical_reports_compare_cleanly(self) -> None:
        from tools.bench import compare_e2e_reports

        result = compare_e2e_reports.compare_reports(report(), report(), required_cases=["cn_short"])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.warnings, [])

    def test_generated_token_mismatch_is_hard_failure(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"][0]["repeats"][0]["generated_token_ids"] = [10, 99, 12]
        result = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertTrue(any(item["code"] == "generated_token_ids_changed" for item in result.failures))

    def test_fixture_identity_change_is_hard_failure(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"][0]["prompt_ids_sha256"] = "8" * 64
        result = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertTrue(any(item["code"] == "case_identity_changed" for item in result.failures))

    def test_stop_token_ids_change_is_hard_failure(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"][0]["stop_token_ids"] = [248046]
        result = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertTrue(any(item["code"] == "case_identity_changed" for item in result.failures))

    def test_chat_identity_changes_are_hard_failures(self) -> None:
        from tools.bench import compare_e2e_reports

        mutations = {
            "prompt_format": lambda value: value["cases"][0].__setitem__(
                "prompt_format", "raw-text"
            ),
            "messages_sha256": lambda value: value["cases"][0].__setitem__(
                "messages_sha256", "8" * 64
            ),
            "rendered_prompt_sha256": lambda value: value["cases"][0].__setitem__(
                "rendered_prompt_sha256", "9" * 64
            ),
            "chat_template_kwargs": lambda value: value["cases"][0].__setitem__(
                "chat_template_kwargs", {"enable_thinking": True}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                candidate = report()
                mutate(candidate)
                result = compare_e2e_reports.compare_reports(report(), candidate)
                self.assertTrue(result.failures)

    def test_chat_template_kwargs_int_bool_is_case_identity_change(self) -> None:
        from tools.bench import compare_e2e_reports

        baseline = report()["cases"][0]
        candidate = report()["cases"][0]
        candidate["chat_template_kwargs"] = {"enable_thinking": 0}
        result = compare_e2e_reports.CompareResult()
        compare_e2e_reports._compare_case_identity(baseline, candidate, result)
        self.assertTrue(any(item["code"] == "case_identity_changed" for item in result.failures))

    def test_missing_q5090_identity_is_hard_failure_by_default(self) -> None:
        from tools.bench import compare_e2e_reports

        baseline = report()
        candidate = report()
        for value in (baseline, candidate):
            del value["weights"]["q5090_path"]
            del value["weights"]["q5090_file_size_bytes"]
            del value["weights"]["q5090_sha256"]
        result = compare_e2e_reports.compare_reports(baseline, candidate)
        self.assertTrue(any(item["code"] == "q5090_identity_missing" for item in result.failures))

    def test_skip_token_check_preserves_case_identity(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"][0]["stop_token_ids"] = [248046]
        candidate["weights"]["q5090_sha256"] = "different"
        candidate["cases"][0]["repeats"][0]["generated_token_ids"] = [10, 99, 12]
        result = compare_e2e_reports.compare_reports(report(), candidate, skip_token_id_check=True)
        self.assertTrue(any(item["code"] == "case_identity_changed" for item in result.failures))
        self.assertFalse(any(item["code"] == "q5090_identity_changed" for item in result.failures))
        self.assertFalse(any(item["code"] == "generated_token_ids_changed" for item in result.failures))

    def test_duplicate_case_names_return_schema_failure(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"].append(candidate["cases"][0])
        result = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertTrue(any(item["code"] == "schema_invalid" for item in result.failures))

    def test_missing_required_cases_are_hard_failures(self) -> None:
        from tools.bench import compare_e2e_reports

        baseline_missing = report()
        baseline_missing["cases"] = [baseline_missing["cases"][0]]
        baseline_result = compare_e2e_reports.compare_reports(
            baseline_missing,
            report(),
            required_cases=["long_2k"],
        )
        self.assertTrue(any(item["code"] == "missing_baseline_case" for item in baseline_result.failures))

        candidate_missing = report()
        candidate_missing["cases"] = [candidate_missing["cases"][0]]
        candidate_result = compare_e2e_reports.compare_reports(
            report(),
            candidate_missing,
            required_cases=["long_2k"],
        )
        self.assertTrue(any(item["code"] == "missing_candidate_case" for item in candidate_result.failures))

    def test_performance_warning_can_be_promoted(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report(tok_s=90.0)
        warned = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertFalse(warned.failures)
        self.assertTrue(any(item["code"] == "throughput_drop" for item in warned.warnings))

        failed = compare_e2e_reports.compare_reports(
            report(),
            candidate,
            fail_on_performance_regression=True,
        )
        self.assertTrue(any(item["code"] == "throughput_drop" for item in failed.failures))

    def test_phase_time_warning(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["cases"][0]["summary"]["prefill_time_s_median"] = 0.03
        result = compare_e2e_reports.compare_reports(report(), candidate)
        self.assertTrue(any(item["code"] == "phase_time_increase" for item in result.warnings))

    def test_memory_warning_requires_percent_and_absolute_threshold(self) -> None:
        from tools.bench import compare_e2e_reports

        baseline = report(workspace_peak=256 * 1024 * 1024)
        candidate = report(workspace_peak=400 * 1024 * 1024)
        result = compare_e2e_reports.compare_reports(baseline, candidate)
        self.assertTrue(any(item["code"] == "memory_peak_increase" for item in result.warnings))

    def test_memory_warning_can_be_promoted(self) -> None:
        from tools.bench import compare_e2e_reports

        baseline = report(workspace_peak=256 * 1024 * 1024)
        candidate = report(workspace_peak=400 * 1024 * 1024)
        result = compare_e2e_reports.compare_reports(
            baseline,
            candidate,
            fail_on_memory_regression=True,
        )
        self.assertTrue(any(item["code"] == "memory_peak_increase" for item in result.failures))

    def test_top_level_policy_changes_are_warnings(self) -> None:
        from tools.bench import compare_e2e_reports

        candidate = report()
        candidate["engine"]["workspace_lifetime_policy"] = "block_scoped_mixer_mlp_rewind"
        candidate["weights"]["load_strategy"] = "streaming_upload"
        candidate["memory"]["accounting_scope"] = "engine_arenas_only"
        result = compare_e2e_reports.compare_reports(report(), candidate)
        codes = {item["code"] for item in result.warnings}
        self.assertIn("workspace_lifetime_policy_changed", codes)
        self.assertIn("load_strategy_changed", codes)
        self.assertIn("memory_accounting_scope_changed", codes)

    def test_cli_writes_json_result(self) -> None:
        from tools.bench import compare_e2e_reports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "compare.json"
            baseline.write_text(json.dumps(report()), encoding="utf-8")
            changed = report(tok_s=90.0)
            candidate.write_text(json.dumps(changed), encoding="utf-8")
            rc = compare_e2e_reports.main(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["artifact_type"], "qus_e2e_compare_result")
            self.assertEqual(value["status"], "warning")

    def test_cli_writes_failure_json_result(self) -> None:
        from tools.bench import compare_e2e_reports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "compare.json"
            baseline.write_text(json.dumps(report()), encoding="utf-8")
            bad = report()
            del bad["weights"]
            candidate.write_text(json.dumps(bad), encoding="utf-8")
            rc = compare_e2e_reports.main(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["artifact_type"], "qus_e2e_compare_result")
            self.assertEqual(value["status"], "fail")


class BaselineSummaryTests(unittest.TestCase):
    def test_make_smoke_summary_extracts_audit_fields(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report()), encoding="utf-8")
            summary = make_baseline_summary.make_summary(report_path, "smoke", decoded_manifest_path=None)
            self.assertEqual(summary["artifact_type"], "qus_e2e_baseline_summary")
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["baseline_class"], "smoke")
            self.assertEqual(summary["source_report_sha256"], common.sha256_file(report_path))
            self.assertEqual(summary["q5090"]["sha256"], "weights-sha")
            self.assertEqual(summary["workspace_lifetime_policy"], "step_reset")
            self.assertFalse(summary["hidden_device_allocations"])
            self.assertEqual([case["name"] for case in summary["cases"]], ["cn_short", "long_2k"])
            self.assertEqual(summary["cases"][0]["prompt_format"], tokenizer_common.PROMPT_FORMAT)
            self.assertEqual(
                summary["cases"][0]["messages_path"],
                "bench/fixtures/prompts/cn_short.messages.json",
            )
            self.assertEqual(summary["cases"][0]["messages_sha256"], SHA_CN_MESSAGES)
            self.assertEqual(summary["cases"][0]["rendered_prompt_sha256"], SHA_CN_RENDERED)
            self.assertTrue(summary["cases"][0]["add_generation_prompt"])
            self.assertFalse(summary["cases"][0]["add_special_tokens"])
            self.assertEqual(
                summary["cases"][0]["chat_template_kwargs"],
                tokenizer_common.CHAT_TEMPLATE_KWARGS,
            )
            self.assertEqual(summary["cases"][0]["stop_token_ids"], tokenizer_common.STOP_TOKEN_IDS)
            self.assertNotIn("eos_token_id", summary["cases"][0])

    def test_smoke_summary_enforces_cn_short_case_and_token_minimum(self) -> None:
        from tools.bench import make_baseline_summary

        def lower_cn_short_requested_tokens(value: dict[str, object]) -> None:
            cn_short = value["cases"][0]
            cn_short["requested_max_new_tokens"] = 7
            cn_short["decode_loop_tokens_requested"] = 6
            cn_short["required_max_context"] = cn_short["prompt_tokens"] + 6

        mutations = {
            "missing_cn_short": lambda value: value["cases"].pop(0),
            "requested_tokens_below_smoke_minimum": lower_cn_short_requested_tokens,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    report_path = root / f"{name}.json"
                    bad = report()
                    mutate(bad)
                    report_path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        make_baseline_summary.make_summary(
                            report_path,
                            "smoke",
                            decoded_manifest_path=None,
                        )

    def test_smoke_summary_rejects_zero_measured_repeats_via_schema(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            bad = report()
            cn_short = bad["cases"][0]
            cn_short["measured_repeats"] = 0
            cn_short["repeats"] = []
            report_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(common.ReportValidationError):
                make_baseline_summary.make_summary(
                    report_path,
                    "smoke",
                    decoded_manifest_path=None,
                )

    def test_summary_rejects_missing_required_audit_fields(self) -> None:
        from tools.bench import make_baseline_summary

        mutations = {
            "run.command": lambda value: value["run"].pop("command"),
            "run.git_commit": lambda value: value["run"].pop("git_commit"),
            "run.worktree_dirty": lambda value: value["run"].pop("worktree_dirty"),
            "weights.q5090_sha256": lambda value: value["weights"].pop("q5090_sha256"),
            "memory.hidden_device_allocations": (
                lambda value: value["memory"].pop("hidden_device_allocations")
            ),
            "engine.workspace_lifetime_policy": (
                lambda value: value["engine"].pop("workspace_lifetime_policy")
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    report_path = Path(tmp) / f"{name.replace('.', '_')}.json"
                    bad = report()
                    mutate(bad)
                    report_path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        make_baseline_summary.make_summary(
                            report_path,
                            "smoke",
                            decoded_manifest_path=None,
                        )

    def test_m3_gate_summary_enforces_required_cases(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            bad = report()
            bad["cases"] = [bad["cases"][0]]
            report_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "m3_gate", decoded_manifest_path=None)

    def test_m3_gate_summary_rejects_missing_actual_repeat_records(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            bad = report()
            bad["cases"][0]["repeats"].pop()
            report_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(common.ReportValidationError):
                make_baseline_summary.make_summary(report_path, "m3_gate", decoded_manifest_path=None)

    def test_m3_gate_summary_rejects_nondeterministic_token_ids(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            bad = report()
            bad["cases"][0]["summary"]["deterministic_token_ids"] = False
            report_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "m3_gate", decoded_manifest_path=None)

    def test_summary_includes_decoded_tokenizer_when_manifest_given(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            decoded_path = root / "decoded_manifest.json"
            report_path.write_text(json.dumps(report()), encoding="utf-8")
            decoded_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "qus_decoded_text_artifacts",
                        "source_report": str(report_path),
                        "readability_gate": "human_smoke_only",
                        "tokenizer": {
                            "tokenizer_source": "local_hf",
                            "tokenizer_model_id": "Qwen/Qwen3.6-27B",
                            "tokenizer_path": "",
                            "tokenizer_json_sha256": "tok",
                            "tokenizer_config_sha256": "cfg",
                            "special_tokens_map_sha256": "special",
                            "chat_template_jinja_sha256": "chat",
                            "generation_config_sha256": "gen",
                        },
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            summary = make_baseline_summary.make_summary(report_path, "smoke", decoded_path)
            self.assertEqual(summary["tokenizer"]["tokenizer_source"], "local_hf")
            self.assertEqual(summary["readability_gate"], "human_smoke_only")
            self.assertEqual(summary["decoded_manifest_path"], str(decoded_path))
            self.assertEqual(summary["decoded_manifest_sha256"], common.sha256_file(decoded_path))

    def test_summary_rejects_unredacted_decoded_manifest(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            decoded_path = root / "decoded_manifest.json"
            report_path.write_text(json.dumps(report()), encoding="utf-8")
            decoded_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "qus_decoded_text_artifacts",
                        "source_report": str(report_path),
                        "readability_gate": "human_smoke_only",
                        "tokenizer": {
                            "tokenizer_source": "local_hf",
                            "tokenizer_model_id": "Qwen/Qwen3.6-27B",
                            "tokenizer_path": "/tmp/local-tokenizer",
                            "tokenizer_json_sha256": "tok",
                            "tokenizer_config_sha256": "cfg",
                            "special_tokens_map_sha256": "special",
                            "chat_template_jinja_sha256": "chat",
                            "generation_config_sha256": "gen",
                        },
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "smoke", decoded_path)

    def test_decoded_manifest_requires_metadata(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report()), encoding="utf-8")

            missing_tokenizer = root / "missing_tokenizer.json"
            missing_tokenizer.write_text(
                json.dumps(
                    {
                        "artifact_type": "qus_decoded_text_artifacts",
                        "readability_gate": "human_smoke_only",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "smoke", missing_tokenizer)

            empty_tokenizer = root / "empty_tokenizer.json"
            empty_tokenizer.write_text(
                json.dumps(
                    {
                        "artifact_type": "qus_decoded_text_artifacts",
                        "readability_gate": "human_smoke_only",
                        "tokenizer": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "smoke", empty_tokenizer)

            missing_gate = root / "missing_gate.json"
            missing_gate.write_text(
                json.dumps(
                    {
                        "artifact_type": "qus_decoded_text_artifacts",
                        "tokenizer": {"tokenizer_source": "local_hf"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                make_baseline_summary.make_summary(report_path, "smoke", missing_gate)

    def test_cli_writes_summary(self) -> None:
        from tools.bench import make_baseline_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            output_path = root / "summary.json"
            report_path.write_text(json.dumps(report()), encoding="utf-8")
            rc = make_baseline_summary.main(
                [
                    "--report",
                    str(report_path),
                    "--output",
                    str(output_path),
                    "--baseline-class",
                    "smoke",
                ]
            )
            self.assertEqual(rc, 0)
            value = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(value["artifact_type"], "qus_e2e_baseline_summary")


if __name__ == "__main__":
    unittest.main()
