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
        "fixture_manifest_sha256": "fixture-sha",
        "prompt_ids_path": f"bench/fixtures/prompts/{name}.ids",
        "prompt_ids_sha256": f"{name}-ids-sha",
        "prompt_tokens": prompt_tokens,
        "requested_max_new_tokens": requested_max_new_tokens,
        "eos_token_id": -1,
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


if __name__ == "__main__":
    unittest.main()
