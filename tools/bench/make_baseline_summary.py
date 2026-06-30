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
        "prefill_prompt_tok_s_median": summary.get("prefill_prompt_tok_s_median"),
        "decode_time_s_median": summary.get("decode_time_s_median"),
        "decode_tok_s_median": summary.get("decode_tok_s_median"),
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
                "prefill_prompt_tok_s_median": case["summary"].get(
                    "prefill_prompt_tok_s_median"
                ),
                "decode_time_s_median": case["summary"].get("decode_time_s_median"),
                "decode_tok_s_median": case["summary"].get("decode_tok_s_median"),
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


def _q5090_sha256(weights: dict[str, Any]) -> str:
    raw = _require_key(weights, "q5090_sha256", "weights")
    if not isinstance(raw, str):
        raise RuntimeError("weights.q5090_sha256 must be a string")
    if raw:
        return raw
    path = _require_key(weights, "q5090_path", "weights")
    if not isinstance(path, str) or not path:
        raise RuntimeError("weights.q5090_path must be a nonempty string")
    q5090_path = Path(path)
    if not q5090_path.exists():
        return ""
    return common.sha256_file(q5090_path)


def _require_decoded_artifact(artifact: dict[str, Any], index: int) -> dict[str, Any]:
    label = f"decoded manifest artifacts[{index}]"
    case_name = _require_key(artifact, "case_name", label)
    if not isinstance(case_name, str) or not case_name:
        raise RuntimeError(f"{label}.case_name must be a nonempty string")
    repeat_index = _require_key(artifact, "repeat_index", label)
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool) or repeat_index < 0:
        raise RuntimeError(f"{label}.repeat_index must be a nonnegative integer")
    clean_text_path = _require_key(artifact, "clean_text_path", label)
    if not isinstance(clean_text_path, str) or not clean_text_path:
        raise RuntimeError(f"{label}.clean_text_path must be a nonempty string")
    clean_chars = _require_key(artifact, "clean_text_chars", label)
    if not isinstance(clean_chars, int) or isinstance(clean_chars, bool) or clean_chars < 0:
        raise RuntimeError(f"{label}.clean_text_chars must be a nonnegative integer")
    clean_sha = _require_key(artifact, "clean_text_sha256", label)
    if not isinstance(clean_sha, str) or len(clean_sha) != 64:
        raise RuntimeError(f"{label}.clean_text_sha256 must be a sha256 string")
    nonempty = _require_key(artifact, "clean_text_nonempty_after_strip", label)
    if not isinstance(nonempty, bool):
        raise RuntimeError(f"{label}.clean_text_nonempty_after_strip must be a boolean")
    return {
        "case_name": case_name,
        "repeat_index": repeat_index,
        "clean_text_path": clean_text_path,
        "clean_text_chars": clean_chars,
        "clean_text_sha256": clean_sha,
        "clean_text_nonempty_after_strip": nonempty,
    }


def _decoded_manifest(
    decoded_manifest_path: Path | None,
    required_nonempty_cases: list[str],
) -> dict[str, Any]:
    if decoded_manifest_path is None:
        if required_nonempty_cases:
            raise RuntimeError(
                "decoded manifest is required for output-validity baseline summaries"
            )
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
    if readability_gate not in common.tokenizer_common.READABILITY_GATES:
        raise RuntimeError("decoded manifest readability_gate is not recognized")
    raw_artifacts = _require_key(manifest, "artifacts", "decoded manifest")
    if not isinstance(raw_artifacts, list):
        raise RuntimeError("decoded manifest artifacts must be a list")
    artifacts = [
        _require_decoded_artifact(common.require_mapping(artifact, "decoded artifact"), index)
        for index, artifact in enumerate(raw_artifacts)
    ]
    for case_name in required_nonempty_cases:
        matches = [
            artifact
            for artifact in artifacts
            if artifact["case_name"] == case_name
            and artifact["clean_text_nonempty_after_strip"] is True
            and artifact["clean_text_chars"] > 0
        ]
        if not matches:
            raise RuntimeError(
                f"decoded manifest must contain nonempty clean output for {case_name}"
            )
    return {
        "tokenizer": tokenizer,
        "readability_gate": readability_gate,
        "decoded_manifest_path": str(decoded_manifest_path),
        "decoded_manifest_sha256": common.sha256_file(decoded_manifest_path),
        "decoded_artifacts": artifacts,
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
        96,
        "cn_short max_new_tokens must be at least 96 for smoke",
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


def _enforce_m3_output_gate(report: dict[str, Any]) -> None:
    cases = common.case_map(report)
    required = ("cn_short", "en_short", "code_short", "math_short")
    for case_name in required:
        if case_name not in cases:
            raise RuntimeError(f"m3_output_gate summary requires {case_name}")
        case = cases[case_name]
        if int(case["requested_max_new_tokens"]) < 96:
            raise RuntimeError(f"{case_name} max_new_tokens must be at least 96 for m3_output_gate")
        if int(case["warmup_repeats"]) < 1 or int(case["measured_repeats"]) < 3:
            raise RuntimeError(f"{case_name} m3_output_gate run requires warmup>=1 and repeats>=3")
        if len(case["repeats"]) < 3:
            raise RuntimeError(
                f"{case_name} m3_output_gate run requires at least 3 measured repeat records"
            )


def _enforce_m3_prefill_gate(report: dict[str, Any]) -> None:
    cases = common.case_map(report)
    if set(cases) != {"long_2k"}:
        raise RuntimeError("m3_prefill_gate summary requires exactly long_2k")
    long_2k = cases["long_2k"]
    if int(long_2k["prompt_tokens"]) < 2048:
        raise RuntimeError("long_2k prompt_tokens must be at least 2048 for m3_prefill_gate")
    if int(long_2k["requested_max_new_tokens"]) != 1:
        raise RuntimeError("long_2k max_new_tokens must be exactly 1 for m3_prefill_gate")
    if int(long_2k["measured_repeats"]) < 1:
        raise RuntimeError("long_2k m3_prefill_gate run requires repeats>=1")
    if len(long_2k["repeats"]) < 1:
        raise RuntimeError("long_2k m3_prefill_gate run requires at least 1 measured repeat record")


def make_summary(
    report_path: Path,
    baseline_class: str,
    decoded_manifest_path: Path | None,
) -> dict[str, Any]:
    if baseline_class not in ("smoke", "m3_output_gate", "m3_prefill_gate"):
        raise RuntimeError("baseline_class must be smoke, m3_output_gate, or m3_prefill_gate")
    report = common.load_json(report_path)
    common.validate_report(report)
    if baseline_class == "smoke":
        _enforce_smoke(report)
    if baseline_class == "m3_output_gate":
        _enforce_m3_output_gate(report)
        _enforce_deterministic_token_ids(report, baseline_class)
    if baseline_class == "m3_prefill_gate":
        _enforce_m3_prefill_gate(report)
        _enforce_deterministic_token_ids(report, baseline_class)
    required_decoded_cases = []
    if baseline_class == "smoke":
        required_decoded_cases = ["cn_short"]
    elif baseline_class == "m3_output_gate":
        required_decoded_cases = ["cn_short", "en_short", "code_short", "math_short"]
    decoded = _decoded_manifest(decoded_manifest_path, required_decoded_cases)
    run = common.require_mapping(report["run"], "run")
    weights = common.require_mapping(report["weights"], "weights")
    memory = common.require_mapping(report["memory"], "memory")
    engine = common.require_mapping(report["engine"], "engine")
    q5090_sha256 = _q5090_sha256(weights)
    if baseline_class in ("m3_output_gate", "m3_prefill_gate") and not q5090_sha256:
        raise RuntimeError(f"{baseline_class} summary requires q5090_sha256")
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
            "sha256": q5090_sha256,
            "conv1d_layout": _require_key(weights, "q5090_conv1d_layout", "weights"),
            "load_strategy": _require_key(weights, "load_strategy", "weights"),
        },
        "cases": [_case_summary(case) for case in report["cases"]],
        "timing_summary": _timing_summary(report),
        "memory_summary": _memory_summary(report),
        "hidden_device_allocations": _require_key(memory, "hidden_device_allocations", "memory"),
        "workspace_lifetime_policy": _require_key(engine, "workspace_lifetime_policy", "engine"),
        "decode_path": _require_key(engine, "decode_path", "engine"),
        "tokenizer": decoded["tokenizer"],
        "readability_gate": decoded["readability_gate"],
        **{
            key: value
            for key, value in decoded.items()
            if key in ("decoded_manifest_path", "decoded_manifest_sha256", "decoded_artifacts")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--baseline-class",
        choices=("smoke", "m3_output_gate", "m3_prefill_gate"),
        required=True,
    )
    parser.add_argument("--decoded-manifest", type=Path)
    args = parser.parse_args(argv)

    summary = make_summary(args.report, args.baseline_class, args.decoded_manifest)
    common.write_json(args.output, summary)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
