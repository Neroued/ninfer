#!/usr/bin/env python3
"""Create committed audit summaries from local raw M2.8 e2e reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench import e2e_report_common as common  # noqa: E402


def _require_key(obj: dict[str, Any], key: str, label: str) -> Any:
    if key not in obj:
        raise RuntimeError(f"{label} missing required field: {key}")
    return obj[key]


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    summary = common.require_mapping(case["summary"], f"{case['name']}.summary")
    return {
        "name": case["name"],
        "fixture_set": case["fixture_set"],
        "fixture_manifest_path": case["fixture_manifest_path"],
        "fixture_manifest_sha256": case["fixture_manifest_sha256"],
        "prompt_format": case["prompt_format"],
        "messages_path": case["messages_path"],
        "messages_sha256": case["messages_sha256"],
        "rendered_prompt_sha256": case["rendered_prompt_sha256"],
        "add_generation_prompt": case["add_generation_prompt"],
        "add_special_tokens": case["add_special_tokens"],
        "chat_template_kwargs": case["chat_template_kwargs"],
        "stop_token_ids": case["stop_token_ids"],
        "prompt_ids_path": case["prompt_ids_path"],
        "prompt_ids_sha256": case["prompt_ids_sha256"],
        "prompt_tokens": case["prompt_tokens"],
        "requested_max_new_tokens": case["requested_max_new_tokens"],
        "warmup_repeats": case["warmup_repeats"],
        "measured_repeats": case["measured_repeats"],
        "deterministic_token_ids": summary.get("deterministic_token_ids"),
        "prefill_time_s_median": summary.get("prefill_time_s_median"),
        "decode_time_s_median": summary.get("decode_time_s_median"),
        "decode_eager_tok_s_median": summary.get("decode_eager_tok_s_median"),
        "e2e_excluding_load_tok_s_median": summary.get("e2e_excluding_load_tok_s_median"),
        "max_weight_arena_peak_used_bytes": summary.get("max_weight_arena_peak_used_bytes"),
        "max_cache_arena_peak_used_bytes": summary.get("max_cache_arena_peak_used_bytes"),
        "max_workspace_arena_peak_used_bytes": summary.get("max_workspace_arena_peak_used_bytes"),
    }


def _timing_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "load_time_s": report["run"].get("load_time_s"),
        "cases": [
            {
                "name": case["name"],
                "prefill_time_s_median": case["summary"].get("prefill_time_s_median"),
                "decode_time_s_median": case["summary"].get("decode_time_s_median"),
                "decode_eager_tok_s_median": case["summary"].get("decode_eager_tok_s_median"),
                "e2e_excluding_load_tok_s_median": case["summary"].get(
                    "e2e_excluding_load_tok_s_median"
                ),
            }
            for case in report["cases"]
        ],
    }


def _memory_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "accounting_scope": report["memory"].get("accounting_scope"),
        "hidden_device_allocations": report["memory"].get("hidden_device_allocations"),
        "arenas": report["memory"].get("arenas", []),
        "cases": [
            {
                "name": case["name"],
                "max_weight_arena_peak_used_bytes": case["summary"].get(
                    "max_weight_arena_peak_used_bytes"
                ),
                "max_cache_arena_peak_used_bytes": case["summary"].get(
                    "max_cache_arena_peak_used_bytes"
                ),
                "max_workspace_arena_peak_used_bytes": case["summary"].get(
                    "max_workspace_arena_peak_used_bytes"
                ),
            }
            for case in report["cases"]
        ],
    }


def _decoded_manifest(decoded_manifest_path: Path | None) -> dict[str, Any]:
    if decoded_manifest_path is None:
        return {"tokenizer": {}, "readability_gate": "not_run"}
    manifest = common.load_json(decoded_manifest_path)
    if manifest.get("artifact_type") != "qus_decoded_text_artifacts":
        raise RuntimeError("decoded manifest artifact_type must be qus_decoded_text_artifacts")
    tokenizer = _require_key(manifest, "tokenizer", "decoded manifest")
    if not isinstance(tokenizer, dict):
        raise RuntimeError("decoded manifest tokenizer must be an object")
    for field in (
        "tokenizer_source",
        "tokenizer_model_id",
        "tokenizer_path",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "special_tokens_map_sha256",
        "chat_template_jinja_sha256",
        "generation_config_sha256",
    ):
        _require_key(tokenizer, field, "decoded manifest tokenizer")
    if tokenizer["tokenizer_path"] != "":
        raise RuntimeError("decoded manifest tokenizer_path must be redacted for committed summaries")
    readability_gate = _require_key(manifest, "readability_gate", "decoded manifest")
    if not isinstance(readability_gate, str):
        raise RuntimeError("decoded manifest readability_gate must be a string")
    return {
        "tokenizer": tokenizer,
        "readability_gate": readability_gate,
        "decoded_manifest_path": str(decoded_manifest_path),
        "decoded_manifest_sha256": common.sha256_file(decoded_manifest_path),
    }


def _require_int_at_least(
    obj: dict[str, Any],
    key: str,
    label: str,
    min_value: int,
    message: str,
) -> None:
    value = _require_key(obj, key, label)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{label}.{key} must be an integer")
    if value < min_value:
        raise RuntimeError(message)


def _enforce_smoke(report: dict[str, Any]) -> None:
    cases = common.case_map(report)
    if "cn_short" not in cases:
        raise RuntimeError("smoke summary requires cn_short")

    cn_short = cases["cn_short"]
    _require_int_at_least(
        cn_short,
        "requested_max_new_tokens",
        "cn_short",
        8,
        "cn_short max_new_tokens must be at least 8 for smoke",
    )
    _require_int_at_least(
        cn_short,
        "measured_repeats",
        "cn_short",
        1,
        "cn_short smoke run requires repeats>=1",
    )


def _enforce_deterministic_token_ids(report: dict[str, Any], baseline_class: str) -> None:
    for case in common.require_list(report["cases"], "cases"):
        case_obj = common.require_mapping(case, "case")
        case_name = str(case_obj["name"])
        summary = common.require_mapping(case_obj["summary"], f"{case_name}.summary")
        if summary.get("deterministic_token_ids") is not True:
            raise RuntimeError(
                f"{baseline_class} summary requires deterministic_token_ids=true for {case_name}"
            )


def _enforce_m3_gate(report: dict[str, Any]) -> None:
    cases = common.case_map(report)
    if "cn_short" not in cases:
        raise RuntimeError("m3_gate summary requires cn_short")
    if "long_2k" not in cases:
        raise RuntimeError("m3_gate summary requires long_2k")

    cn_short = cases["cn_short"]
    long_2k = cases["long_2k"]
    if int(cn_short["requested_max_new_tokens"]) < 128:
        raise RuntimeError("cn_short max_new_tokens must be at least 128 for m3_gate")
    if int(cn_short["warmup_repeats"]) < 1 or int(cn_short["measured_repeats"]) < 3:
        raise RuntimeError("cn_short m3_gate run requires warmup>=1 and repeats>=3")
    if len(cn_short["repeats"]) < 3:
        raise RuntimeError("cn_short m3_gate run requires at least 3 measured repeat records")
    if int(long_2k["prompt_tokens"]) < 2048:
        raise RuntimeError("long_2k prompt_tokens must be at least 2048 for m3_gate")
    if int(long_2k["measured_repeats"]) < 1:
        raise RuntimeError("long_2k m3_gate run requires repeats>=1")
    if len(long_2k["repeats"]) < 1:
        raise RuntimeError("long_2k m3_gate run requires at least 1 measured repeat record")
    if not report["weights"].get("q5090_sha256"):
        raise RuntimeError("m3_gate summary requires q5090_sha256")


def make_summary(
    report_path: Path,
    baseline_class: str,
    decoded_manifest_path: Path | None,
) -> dict[str, Any]:
    if baseline_class not in ("smoke", "m3_gate"):
        raise RuntimeError("baseline_class must be smoke or m3_gate")
    report = common.load_json(report_path)
    common.validate_report(report)
    if baseline_class == "smoke":
        _enforce_smoke(report)
    if baseline_class == "m3_gate":
        _enforce_m3_gate(report)
        _enforce_deterministic_token_ids(report, baseline_class)
    decoded = _decoded_manifest(decoded_manifest_path)
    run = common.require_mapping(report["run"], "run")
    weights = common.require_mapping(report["weights"], "weights")
    memory = common.require_mapping(report["memory"], "memory")
    engine = common.require_mapping(report["engine"], "engine")
    return {
        "artifact_type": "qus_e2e_baseline_summary",
        "schema_version": 1,
        "baseline_class": baseline_class,
        "source_report_path": str(report_path),
        "source_report_sha256": common.sha256_file(report_path),
        "command": _require_key(run, "command", "run"),
        "git_commit": _require_key(run, "git_commit", "run"),
        "worktree_dirty": _require_key(run, "worktree_dirty", "run"),
        "q5090": {
            "path": _require_key(weights, "q5090_path", "weights"),
            "file_size_bytes": _require_key(weights, "q5090_file_size_bytes", "weights"),
            "sha256": _require_key(weights, "q5090_sha256", "weights"),
            "conv1d_layout": _require_key(weights, "q5090_conv1d_layout", "weights"),
            "load_strategy": _require_key(weights, "load_strategy", "weights"),
        },
        "cases": [_case_summary(case) for case in report["cases"]],
        "timing_summary": _timing_summary(report),
        "memory_summary": _memory_summary(report),
        "hidden_device_allocations": _require_key(memory, "hidden_device_allocations", "memory"),
        "workspace_lifetime_policy": _require_key(engine, "workspace_lifetime_policy", "engine"),
        "tokenizer": decoded["tokenizer"],
        "readability_gate": decoded["readability_gate"],
        **{
            key: value
            for key, value in decoded.items()
            if key in ("decoded_manifest_path", "decoded_manifest_sha256")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-class", choices=("smoke", "m3_gate"), required=True)
    parser.add_argument("--decoded-manifest", type=Path)
    args = parser.parse_args(argv)

    summary = make_summary(args.report, args.baseline_class, args.decoded_manifest)
    common.write_json(args.output, summary)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
