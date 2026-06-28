#!/usr/bin/env python3
"""Decode generated token ids from an e2e report into sidecar text artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bench import tokenizer_common as common


def _default_output_dir(report_path: Path) -> Path:
    return report_path.with_suffix(".decoded")


def _require_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise RuntimeError("expected schema_version 1")
    if report.get("artifact_type") != "qus_e2e_benchmark_report":
        raise RuntimeError("expected qus_e2e_benchmark_report")
    if report.get("status") != "ok":
        raise RuntimeError("decode sidecars require an ok e2e report")
    if not isinstance(report.get("cases"), list):
        raise RuntimeError("report cases must be a list")


def _case_chat_metadata(case: dict[str, Any], case_name: str) -> dict[str, Any]:
    if "eos_token_id" in case:
        raise RuntimeError(
            f"case {case_name} uses stop_token_ids; eos_token_id is not allowed"
        )
    required = (
        "prompt_format",
        "messages_path",
        "messages_sha256",
        "rendered_prompt_sha256",
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template_kwargs",
        "stop_token_ids",
    )
    missing = [field for field in required if field not in case]
    if missing:
        raise RuntimeError(f"case {case_name} missing chat identity fields: {', '.join(missing)}")
    if case["prompt_format"] != common.PROMPT_FORMAT:
        raise RuntimeError(f"case {case_name} prompt_format must be {common.PROMPT_FORMAT}")
    messages_path = case["messages_path"]
    if (
        not isinstance(messages_path, str)
        or not messages_path
        or not messages_path.endswith(common.MESSAGE_FILE_SUFFIX)
    ):
        raise RuntimeError(
            f"case {case_name} messages_path must be nonempty and end with "
            f"{common.MESSAGE_FILE_SUFFIX}"
        )
    for field in ("messages_sha256", "rendered_prompt_sha256"):
        value = case[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"case {case_name} {field} must be a 64-char lowercase hex sha256")
    if case["add_generation_prompt"] is not common.ADD_GENERATION_PROMPT:
        raise RuntimeError(
            f"case {case_name} add_generation_prompt must be {common.ADD_GENERATION_PROMPT}"
        )
    if case["add_special_tokens"] is not common.ADD_SPECIAL_TOKENS:
        raise RuntimeError(
            f"case {case_name} add_special_tokens must be {common.ADD_SPECIAL_TOKENS}"
        )
    chat_template_kwargs = case["chat_template_kwargs"]
    if (
        not isinstance(chat_template_kwargs, dict)
        or set(chat_template_kwargs) != {"enable_thinking"}
        or chat_template_kwargs["enable_thinking"] is not False
    ):
        raise RuntimeError(
            f"case {case_name} chat_template_kwargs must equal {common.CHAT_TEMPLATE_KWARGS}"
        )
    stop_token_ids = case["stop_token_ids"]
    if (
        not isinstance(stop_token_ids, list)
        or not stop_token_ids
        or not all(_is_nonnegative_int(value) for value in stop_token_ids)
    ):
        raise RuntimeError(f"case {case_name} stop_token_ids must be nonempty nonnegative ints")
    return {field: case[field] for field in required}


def _require_common_chat_metadata(
    current: dict[str, Any] | None,
    next_value: dict[str, Any],
    case_name: str,
) -> dict[str, Any]:
    fields = (
        "prompt_format",
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template_kwargs",
        "stop_token_ids",
    )
    common_value = {field: next_value[field] for field in fields}
    if current is None:
        return common_value
    changed = {
        field: (current[field], common_value[field])
        for field in fields
        if current[field] != common_value[field]
    }
    if changed:
        raise RuntimeError(f"case {case_name} chat identity differs from earlier cases: {changed}")
    return current


def _repeat_index(repeat: dict[str, Any], fallback: int, case_name: str) -> int:
    if "repeat_index" not in repeat:
        return fallback
    value = repeat["repeat_index"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"case {case_name} repeat_index must be a nonnegative integer")
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def decode_report(
    report_path: Path,
    tokenizer: Any,
    tokenizer_metadata: dict[str, str],
    output_dir: Path | None,
) -> dict[str, Any]:
    source_report = str(report_path)
    report = common.read_json(report_path)
    _require_report(report)
    out_dir = output_dir if output_dir is not None else _default_output_dir(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    common_chat_metadata: dict[str, Any] | None = None
    for case_index, case in enumerate(report["cases"]):
        case_name = str(case.get("name", f"case{case_index}"))
        chat_metadata = _case_chat_metadata(case, case_name)
        common_chat_metadata = _require_common_chat_metadata(
            common_chat_metadata,
            chat_metadata,
            case_name,
        )
        repeats = case.get("repeats")
        if not isinstance(repeats, list):
            raise RuntimeError(f"case {case_name} repeats must be a list")
        case_dir = out_dir / f"case{case_index}_{common.safe_name(case_name)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for repeat in repeats:
            if not isinstance(repeat, dict):
                raise RuntimeError(f"case {case_name} repeat must be an object")
            repeat_index = _repeat_index(repeat, len(artifacts), case_name)
            ids = repeat.get("generated_token_ids")
            if not isinstance(ids, list) or not ids or not all(_is_nonnegative_int(v) for v in ids):
                raise RuntimeError(f"case {case_name} repeat {repeat_index} has invalid token ids")
            raw_decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
            clean_decoded = tokenizer.decode(list(ids), skip_special_tokens=True)
            raw_path = case_dir / f"repeat_{repeat_index}.raw.txt"
            clean_path = case_dir / f"repeat_{repeat_index}.clean.txt"
            raw_path.write_text(raw_decoded, encoding="utf-8")
            clean_path.write_text(clean_decoded, encoding="utf-8")
            artifacts.append(
                {
                    "case_index": case_index,
                    "case_name": case_name,
                    "repeat_index": repeat_index,
                    "raw_text_path": str(raw_path),
                    "clean_text_path": str(clean_path),
                    "raw_text_chars": len(raw_decoded),
                    "clean_text_chars": len(clean_decoded),
                    "clean_text_sha256": common.sha256_text(clean_decoded),
                    "clean_text_nonempty_after_strip": bool(clean_decoded.strip()),
                    "generated_tokens_total": len(ids),
                    **chat_metadata,
                }
            )

    if not artifacts:
        raise RuntimeError("report contains no generated token ids to decode")
    if common_chat_metadata is None:
        raise RuntimeError("report contains no cases to decode")

    manifest = {
        "artifact_type": "qus_decoded_text_artifacts",
        "source_report": source_report,
        "readability_gate": "human_smoke_only",
        "tokenizer": tokenizer_metadata,
        **common_chat_metadata,
        "artifacts": artifacts,
    }
    common.write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    tokenizer_path = common.resolve_tokenizer_path(args.tokenizer_path)
    tokenizer = common.load_tokenizer(tokenizer_path)
    metadata = common.tokenizer_metadata(tokenizer_path, redact_path=True)
    manifest = decode_report(args.report, tokenizer, metadata, args.output_dir)
    manifest_path = (
        args.output_dir if args.output_dir is not None else args.report.with_suffix(".decoded")
    ) / "manifest.json"
    print(f"wrote {manifest_path}")
    print(f"decoded {len(manifest['artifacts'])} repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
