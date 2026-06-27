#!/usr/bin/env python3
"""Shared helpers for M2.8 benchmark fixture and decode tooling."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

TOKENIZER_MODEL_ID = "Qwen/Qwen3.6-27B"
FIXTURE_SET = "m2.8-v1"
REQUIRED_CASES = ("cn_short", "en_short", "code_short", "math_short", "long_2k")
READABILITY_GATES = ("human_smoke_only", "not_run")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_tokenizer_path(cli_path: str | None) -> Path:
    raw = cli_path or os.environ.get("QUS_TOKENIZER_PATH")
    if not raw:
        raise RuntimeError(
            "tokenizer path required; pass --tokenizer-path or set QUS_TOKENIZER_PATH"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"tokenizer path is not a directory: {path}")
    return path


def _optional_hash(root: Path, name: str) -> str:
    path = root / name
    return sha256_file(path) if path.exists() else ""


def tokenizer_metadata(tokenizer_path: Path, redact_path: bool = False) -> dict[str, str]:
    return {
        "tokenizer_source": "local_hf",
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "tokenizer_path": "" if redact_path else str(tokenizer_path),
        "tokenizer_json_sha256": _optional_hash(tokenizer_path, "tokenizer.json"),
        "tokenizer_config_sha256": _optional_hash(tokenizer_path, "tokenizer_config.json"),
        "special_tokens_map_sha256": _optional_hash(tokenizer_path, "special_tokens_map.json"),
    }


def load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenizer-dependent tools; install "
            "tools/bench/requirements.txt in the active Python environment"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def parse_ids_text(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        raise ValueError(".ids content is empty")
    ids: list[int] = []
    for token in stripped.split():
        if not re.fullmatch(r"[0-9]+", token):
            raise ValueError(f"invalid token id in .ids content: {token!r}")
        ids.append(int(token))
    return ids


def format_ids(ids: Iterable[int], per_line: int = 32) -> str:
    values = list(ids)
    if not values:
        raise ValueError("cannot write empty .ids content")
    for value in values:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"token ids must be nonnegative ints, got {value!r}")
    lines = []
    for i in range(0, len(values), per_line):
        lines.append(" ".join(str(v) for v in values[i : i + per_line]))
    return "\n".join(lines) + "\n"


def read_ids(path: Path) -> list[int]:
    return parse_ids_text(path.read_text(encoding="utf-8"))


def write_ids(path: Path, ids: Iterable[int]) -> None:
    path.write_text(format_ids(ids), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "case"
