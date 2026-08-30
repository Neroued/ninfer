#!/usr/bin/env python3
"""Build or verify the fixed NInfer causal-perplexity corpus.

Generation consumes explicit local snapshots and a tokenizer used only to size streams. It never
downloads data. Verification consumes only the committed corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CORPUS_ID = "ninfer-ppl-1m-v1"
SIZING_TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
APPROXIMATE_STREAM_TOKENS = 65_536
STREAMS_PER_DOMAIN = 4

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"
WIKITEXT_FILE = Path("wikitext-2-raw-v1/test-00000-of-00001.parquet")
WIKITEXT_SHA256 = "3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925"

PG19_REPO = "deepmind/pg19"
PG19_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
PG19_TEST_FILES = Path("data/test_files.txt")
PG19_TEST_FILES_SHA256 = "c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c"
PG19_METADATA_SHA256 = "fbb2fdb48522927b2e16aa52950f2afeb83c6fa8fed45f0c3dd834e9bc9b43b9"
PG19_SELECTED_SHA256 = {
    "34467": "32d9cc33391e172ebaf885e574b365ee9738fa0041537a5c8a9859e7d84b6181",
    "35246": "4362445794452e6df50919837f9aa851539a11eaed673b8eb3ed613ac0b1a84b",
    "36256": "84ec4641bc67f1c91c19be710e42f6009dec808e5f174fede2ef65d00e5d2a92",
    "54624": "8c7c0d63e4253003d93727e5d9dfe1245ecfef2ac71fc4b48969f685d150e6e8",
    "3129": "7e561c6b6d4fa20f508b9484f56a9703a88d469e7c2d7a47d91e74431e4c6ff0",
    "30752": "14f951b46330ff912bf29fe4888d668fc4693cd05b9d87f0c2fd0d985f3c29de",
    "46915": "4786e48c791bb791957b629eee30de504a8e03fb35fa0092bef278c8ed10ee83",
    "8559": "eed15b11e09b418eee4e8b38bd324b805934abbb22c340c7561514a89be4b981",
}

WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_REVISION = "f92acb9c45af8e9c752a89bdece1594a47d233d4"
WIKIPEDIA_FILE = Path("20231101.zh/train-00002-of-00006.parquet")
WIKIPEDIA_SHA256 = "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"
WIKIPEDIA_HASH_CANDIDATES = 4096

NINFER_REVISION = "f8fc0f5098d7b16ffa85a681b666b8fcec50efd7"
NINFER_SUFFIXES = frozenset({".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".cuh"})
NINFER_ROOTS = ("include/ninfer/", "src/", "apps/")

DOMAIN_LAYOUT = (
    ("english_reference", "wikitext"),
    ("english_long_form", "pg19"),
    ("chinese_reference", "zhwiki"),
    ("ninfer_code", "ninfer"),
)


@dataclass(frozen=True)
class Unit:
    text: str
    source_id: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class BuiltStream:
    stream_id: str
    domain: str
    text: str
    sizing_tokens: int
    segments: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} sha256 mismatch: expected {expected}, got {actual}: {path}")


def normalized_utf8(data: bytes, label: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{label} is not valid UTF-8: {exc}") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def ensure_lf_suffix(text: str, count: int) -> str:
    present = len(text) - len(text.rstrip("\n"))
    return text + "\n" * max(0, count - present)


def seeded_order_key(domain: str, source_id: str) -> tuple[bytes, str]:
    material = f"{CORPUS_ID}:{domain}:{source_id}".encode("utf-8")
    return hashlib.sha256(material).digest(), source_id


def make_units(text: str, source_id: str, boundary: str) -> list[Unit]:
    if boundary == "record":
        pieces = [text]
    elif boundary == "line":
        pieces = text.splitlines(keepends=True)
    elif boundary == "paragraph":
        pieces = []
        pending: list[str] = []
        for line in text.splitlines(keepends=True):
            pending.append(line)
            if line.strip() == "":
                pieces.append("".join(pending))
                pending.clear()
        if pending:
            pieces.append("".join(pending))
    else:
        raise AssertionError(f"unsupported natural boundary: {boundary}")

    units: list[Unit] = []
    byte_cursor = 0
    for piece in pieces:
        byte_count = len(piece.encode("utf-8"))
        if piece:
            units.append(Unit(piece, source_id, byte_cursor, byte_cursor + byte_count))
        byte_cursor += byte_count
    return units


def load_tokenizer(path: Path) -> Any:
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    require_file_hash(
        tokenizer_json, SIZING_TOKENIZER_JSON_SHA256, "sizing tokenizer.json"
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("tokenizers is required to prepare the PPL corpus") from exc
    return Tokenizer.from_file(str(tokenizer_json))


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def coalesce_segments(units: Sequence[Unit]) -> tuple[dict[str, Any], ...]:
    segments: list[dict[str, Any]] = []
    for unit in units:
        if (
            segments
            and segments[-1]["source_id"] == unit.source_id
            and segments[-1]["end_byte"] == unit.start_byte
        ):
            segments[-1]["end_byte"] = unit.end_byte
        else:
            segments.append(
                {
                    "source_id": unit.source_id,
                    "start_byte": unit.start_byte,
                    "end_byte": unit.end_byte,
                }
            )
    return tuple(segments)


def partition_domain(
    tokenizer: Any,
    domain: str,
    stream_prefix: str,
    units: Sequence[Unit],
) -> list[BuiltStream]:
    if not units:
        raise SystemExit(f"{domain}: source produced no text units")
    estimated = [token_count(tokenizer, unit.text) for unit in units]
    cursor = 0
    streams: list[BuiltStream] = []
    for stream_index in range(STREAMS_PER_DOMAIN):
        estimate = 0
        end = cursor
        while end < len(units) and estimate < APPROXIMATE_STREAM_TOKENS:
            estimate += estimated[end]
            end += 1
        if end == cursor:
            raise SystemExit(f"{domain}: no source content remains for stream {stream_index}")

        candidate_ends = {end}
        if end - 1 > cursor:
            candidate_ends.add(end - 1)
        candidates: list[tuple[int, int, int, str]] = []
        for candidate_end in sorted(candidate_ends):
            candidate_text = "".join(unit.text for unit in units[cursor:candidate_end])
            candidate_tokens = token_count(tokenizer, candidate_text)
            candidates.append(
                (
                    abs(candidate_tokens - APPROXIMATE_STREAM_TOKENS),
                    1 if candidate_tokens < APPROXIMATE_STREAM_TOKENS else 0,
                    candidate_end,
                    candidate_text,
                )
            )
        _, _, chosen_end, text = min(candidates, key=lambda item: item[:3])
        sizing_tokens = token_count(tokenizer, text)
        if sizing_tokens < 2:
            raise SystemExit(f"{domain}: stream {stream_index} has fewer than two tokens")
        selected = units[cursor:chosen_end]
        streams.append(
            BuiltStream(
                stream_id=f"{stream_prefix}-{stream_index:02d}",
                domain=domain,
                text=text,
                sizing_tokens=sizing_tokens,
                segments=coalesce_segments(selected),
            )
        )
        cursor = chosen_end
    return streams


def wikitext_units(snapshot: Path) -> tuple[list[Unit], dict[str, Any]]:
    source = snapshot / WIKITEXT_FILE
    require_file_hash(source, WIKITEXT_SHA256, "WikiText-2 raw test parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to prepare the PPL corpus") from exc
    values = pq.ParquetFile(source).read(columns=["text"]).column("text").to_pylist()
    units: list[Unit] = []
    for row, value in enumerate(values):
        if not isinstance(value, str):
            raise SystemExit(f"WikiText row {row} is not text")
        text = ensure_lf_suffix(value.replace("\r\n", "\n").replace("\r", "\n"), 1)
        units.extend(make_units(text, f"{WIKITEXT_FILE.as_posix()}#row={row}", "record"))
    source_info = {
        "repository": WIKITEXT_REPO,
        "revision": WIKITEXT_REVISION,
        "split": "wikitext-2-raw-v1/test",
        "input": WIKITEXT_FILE.as_posix(),
        "input_sha256": WIKITEXT_SHA256,
    }
    return units, source_info


def pg19_metadata(path: Path) -> dict[str, dict[str, str]]:
    require_file_hash(path, PG19_METADATA_SHA256, "PG-19 metadata.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            handle,
            fieldnames=["id", "short_book_title", "publication_date", "url"],
        )
        return {row["id"]: row for row in reader}


def pg19_units(snapshot: Path, assets: Path) -> tuple[list[Unit], dict[str, Any]]:
    test_files = snapshot / PG19_TEST_FILES
    require_file_hash(test_files, PG19_TEST_FILES_SHA256, "PG-19 pinned test file list")
    listed = [Path(line).stem for line in test_files.read_text(encoding="utf-8").splitlines()]
    ordered = sorted(listed, key=lambda source_id: seeded_order_key("pg19", source_id))
    selected = list(PG19_SELECTED_SHA256)
    if ordered[: len(selected)] != selected:
        raise SystemExit("PG-19 selected ids no longer match the fixed seeded order")

    metadata = pg19_metadata(assets / "metadata.csv")
    units: list[Unit] = []
    selected_metadata: list[dict[str, Any]] = []
    for book_id in selected:
        source = assets / "test" / f"{book_id}.txt"
        require_file_hash(source, PG19_SELECTED_SHA256[book_id], f"PG-19 book {book_id}")
        text = ensure_lf_suffix(normalized_utf8(source.read_bytes(), f"PG-19 book {book_id}"), 2)
        units.extend(make_units(text, f"test/{book_id}.txt", "paragraph"))
        row = metadata.get(book_id)
        if row is None:
            raise SystemExit(f"PG-19 metadata is missing book {book_id}")
        selected_metadata.append(
            {
                "id": book_id,
                "title": row["short_book_title"],
                "publication_date": int(row["publication_date"]),
                "url": row["url"],
                "sha256": PG19_SELECTED_SHA256[book_id],
            }
        )
    source_info = {
        "repository": PG19_REPO,
        "revision": PG19_REVISION,
        "split": "test",
        "test_files_sha256": PG19_TEST_FILES_SHA256,
        "metadata_sha256": PG19_METADATA_SHA256,
        "selected_books": selected_metadata,
    }
    return units, source_info


def wikipedia_units(snapshot: Path) -> tuple[list[Unit], dict[str, Any]]:
    source = snapshot / WIKIPEDIA_FILE
    require_file_hash(source, WIKIPEDIA_SHA256, "Chinese Wikipedia parquet shard")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to prepare the PPL corpus") from exc
    parquet = pq.ParquetFile(source)
    positions: list[tuple[tuple[bytes, str], str, int, int]] = []
    for row_group in range(parquet.num_row_groups):
        ids = parquet.read_row_group(row_group, columns=["id"]).column("id").to_pylist()
        for row, article_id in enumerate(ids):
            article_id = str(article_id)
            positions.append(
                (seeded_order_key("zhwiki", article_id), article_id, row_group, row)
            )
    positions.sort(key=lambda item: item[0])
    selected = positions[:WIKIPEDIA_HASH_CANDIDATES]
    wanted_by_group: dict[int, dict[int, str]] = {}
    for _, article_id, row_group, row in selected:
        wanted_by_group.setdefault(row_group, {})[row] = article_id

    records: dict[str, tuple[str, str]] = {}
    for row_group, wanted in wanted_by_group.items():
        table = parquet.read_row_group(row_group, columns=["id", "title", "text"])
        ids = table.column("id")
        titles = table.column("title")
        texts = table.column("text")
        for row, expected_id in wanted.items():
            actual_id = str(ids[row].as_py())
            if actual_id != expected_id:
                raise SystemExit("Chinese Wikipedia row identity changed while reading parquet")
            records[actual_id] = (str(titles[row].as_py()), str(texts[row].as_py()))

    units: list[Unit] = []
    for _, article_id, _, _ in selected:
        title, raw_text = records[article_id]
        text = ensure_lf_suffix(raw_text.replace("\r\n", "\n").replace("\r", "\n"), 2)
        units.extend(make_units(text, f"article/{article_id}", "paragraph"))
        if not title:
            raise SystemExit(f"Chinese Wikipedia article {article_id} has no title")
    source_info = {
        "repository": WIKIPEDIA_REPO,
        "revision": WIKIPEDIA_REVISION,
        "config": "20231101.zh",
        "input": WIKIPEDIA_FILE.as_posix(),
        "input_sha256": WIKIPEDIA_SHA256,
        "article_url_template": "https://zh.wikipedia.org/?curid={article_id}",
        "selection": "first 4096 article ids in ninfer-ppl-1m-v1 seeded hash order",
    }
    return units, source_info


def run_git(repo: Path, arguments: Sequence[str], *, text: bool) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        error = result.stderr.strip() if text else result.stderr.decode("utf-8", errors="replace")
        raise SystemExit(f"git {' '.join(arguments)} failed: {error}")
    return result.stdout


def ninfer_units(repo: Path) -> tuple[list[Unit], dict[str, Any]]:
    run_git(repo, ["cat-file", "-e", f"{NINFER_REVISION}^{{commit}}"], text=True)
    listing = run_git(repo, ["ls-tree", "-r", "--name-only", NINFER_REVISION], text=True)
    assert isinstance(listing, str)
    paths = [
        path
        for path in listing.splitlines()
        if path.startswith(NINFER_ROOTS) and Path(path).suffix in NINFER_SUFFIXES
    ]
    paths.sort(key=lambda path: seeded_order_key("ninfer", path))
    units: list[Unit] = []
    for path in paths:
        raw = run_git(repo, ["show", f"{NINFER_REVISION}:{path}"], text=False)
        assert isinstance(raw, bytes)
        text = ensure_lf_suffix(normalized_utf8(raw, f"NInfer {path}"), 1)
        units.extend(make_units(text, path, "line"))
    source_info = {
        "repository": "Neroued/ninfer",
        "revision": NINFER_REVISION,
        "roots": list(NINFER_ROOTS),
        "suffixes": sorted(NINFER_SUFFIXES),
        "selection": "project-owned production files in fixed seeded path order",
    }
    return units, source_info


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(args: argparse.Namespace, tokenizer: Any) -> None:
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must not exist or must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_loaders = {
        "wikitext": lambda: wikitext_units(args.wikitext_snapshot),
        "pg19": lambda: pg19_units(args.pg19_snapshot, args.pg19_assets),
        "zhwiki": lambda: wikipedia_units(args.wikipedia_snapshot),
        "ninfer": lambda: ninfer_units(args.ninfer_repo),
    }
    all_streams: list[BuiltStream] = []
    provenance: dict[str, list[dict[str, Any]]] = {}
    for domain, prefix in DOMAIN_LAYOUT:
        print(f"[corpus] loading domain={domain}", file=sys.stderr)
        units, source_info = source_loaders[prefix]()
        streams = partition_domain(tokenizer, domain, prefix, units)
        all_streams.extend(streams)
        provenance[prefix] = [
            {
                "stream_id": stream.stream_id,
                "domain": stream.domain,
                "source": source_info,
                "segments": list(stream.segments),
            }
            for stream in streams
        ]
        print(
            f"[corpus] ready domain={domain} sizing_token_range="
            f"{min(stream.sizing_tokens for stream in streams)}.."
            f"{max(stream.sizing_tokens for stream in streams)}",
            file=sys.stderr,
        )

    stream_entries: list[dict[str, Any]] = []
    for stream in all_streams:
        relative = Path("data") / stream.stream_id.split("-")[0] / f"{stream.stream_id[-2:]}.txt"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = stream.text.encode("utf-8")
        destination.write_bytes(encoded)
        stream_entries.append(
            {
                "id": stream.stream_id,
                "domain": stream.domain,
                "path": relative.as_posix(),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )

    provenance_dir = output / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        provenance_dir / "generation.json",
        {
            "corpus_id": CORPUS_ID,
            "sizing": {
                "role": "approximate stream boundary selection only",
                "tokenizer": "Qwen3.6 family tokenizer.json",
                "tokenizer_json_sha256": SIZING_TOKENIZER_JSON_SHA256,
                "add_special_tokens": False,
                "approximate_tokens_per_stream": APPROXIMATE_STREAM_TOKENS,
            },
        },
    )
    for prefix, entries in provenance.items():
        with (provenance_dir / f"{prefix}.jsonl").open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    full = [entry["id"] for entry in stream_entries]
    quick = [f"{prefix}-00" for _, prefix in DOMAIN_LAYOUT]
    manifest = {
        "corpus_id": CORPUS_ID,
        "streams": stream_entries,
        "modes": {"quick": quick, "full": full},
    }
    write_json(output / "manifest.json", manifest)
    verify(output)


def require_manifest_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def verify(output: Path) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing corpus manifest: {manifest_path}")
    manifest = require_manifest_dict(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    if set(manifest) != {"corpus_id", "streams", "modes"}:
        raise SystemExit("manifest fields do not match the tokenizer-independent corpus contract")
    if manifest.get("corpus_id") != CORPUS_ID:
        raise SystemExit(f"unexpected corpus_id: {manifest.get('corpus_id')!r}")
    streams = manifest.get("streams")
    if not isinstance(streams, list) or len(streams) != len(DOMAIN_LAYOUT) * STREAMS_PER_DOMAIN:
        raise SystemExit("manifest must contain exactly 16 streams")
    expected_ids = [
        f"{prefix}-{index:02d}"
        for _, prefix in DOMAIN_LAYOUT
        for index in range(STREAMS_PER_DOMAIN)
    ]
    actual_ids = [entry.get("id") for entry in streams if isinstance(entry, dict)]
    if actual_ids != expected_ids:
        raise SystemExit("manifest stream ids or order do not match the fixed corpus contract")

    total_bytes = 0
    domain_by_prefix = {prefix: domain for domain, prefix in DOMAIN_LAYOUT}
    for stream_index, entry in enumerate(streams):
        entry = require_manifest_dict(entry, "stream entry")
        if set(entry) != {"id", "domain", "path", "bytes", "sha256"}:
            raise SystemExit(
                "stream fields do not match the tokenizer-independent corpus contract"
            )
        expected_id = expected_ids[stream_index]
        prefix = expected_id.rsplit("-", 1)[0]
        if entry.get("domain") != domain_by_prefix[prefix]:
            raise SystemExit(f"stream {expected_id} domain mismatch")
        path_text = entry.get("path")
        if not isinstance(path_text, str):
            raise SystemExit(f"stream {entry.get('id')} path must be a string")
        expected_path = f"data/{prefix}/{expected_id[-2:]}.txt"
        if path_text != expected_path:
            raise SystemExit(f"stream {expected_id} path must be {expected_path}")
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"stream {entry.get('id')} path must remain inside the corpus")
        path = output / relative
        data = path.read_bytes()
        if b"\r" in data:
            raise SystemExit(f"stream {entry.get('id')} contains a non-LF line ending")
        if b"\0" in data:
            raise SystemExit(f"stream {entry.get('id')} contains a NUL byte")
        text = normalized_utf8(data, f"stream {entry.get('id')}")
        if not text:
            raise SystemExit(f"stream {entry.get('id')} is empty")
        if entry.get("bytes") != len(data):
            raise SystemExit(f"stream {entry.get('id')} byte count mismatch")
        digest = hashlib.sha256(data).hexdigest()
        if entry.get("sha256") != digest:
            raise SystemExit(f"stream {entry.get('id')} sha256 mismatch")
        total_bytes += len(data)

    stream_bytes = {entry["id"]: entry["bytes"] for entry in streams}

    modes = require_manifest_dict(manifest.get("modes"), "manifest modes")
    expected_quick = [f"{prefix}-00" for _, prefix in DOMAIN_LAYOUT]
    if modes.get("quick") != expected_quick or modes.get("full") != expected_ids:
        raise SystemExit("manifest quick/full modes do not match the fixed corpus contract")

    provenance_dir = output / "provenance"
    generation = require_manifest_dict(
        json.loads((provenance_dir / "generation.json").read_text(encoding="utf-8")),
        "generation provenance",
    )
    expected_generation = {
        "corpus_id": CORPUS_ID,
        "sizing": {
            "role": "approximate stream boundary selection only",
            "tokenizer": "Qwen3.6 family tokenizer.json",
            "tokenizer_json_sha256": SIZING_TOKENIZER_JSON_SHA256,
            "add_special_tokens": False,
            "approximate_tokens_per_stream": APPROXIMATE_STREAM_TOKENS,
        },
    }
    if generation != expected_generation:
        raise SystemExit("generation provenance does not match the fixed corpus build record")

    expected_sources = {
        "wikitext": {
            "repository": WIKITEXT_REPO,
            "revision": WIKITEXT_REVISION,
            "input_sha256": WIKITEXT_SHA256,
        },
        "pg19": {
            "repository": PG19_REPO,
            "revision": PG19_REVISION,
            "test_files_sha256": PG19_TEST_FILES_SHA256,
            "metadata_sha256": PG19_METADATA_SHA256,
        },
        "zhwiki": {
            "repository": WIKIPEDIA_REPO,
            "revision": WIKIPEDIA_REVISION,
            "input_sha256": WIKIPEDIA_SHA256,
            "article_url_template": "https://zh.wikipedia.org/?curid={article_id}",
        },
        "ninfer": {
            "repository": "Neroued/ninfer",
            "revision": NINFER_REVISION,
        },
    }
    for _, prefix in DOMAIN_LAYOUT:
        path = provenance_dir / f"{prefix}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != STREAMS_PER_DOMAIN:
            raise SystemExit(f"{path} must contain four stream records")
        ids = []
        for line in lines:
            record = require_manifest_dict(json.loads(line), f"{path} record")
            if set(record) != {"stream_id", "domain", "source", "segments"}:
                raise SystemExit(f"{path}: record fields do not match the provenance contract")
            ids.append(record.get("stream_id"))
            if record.get("domain") != domain_by_prefix[prefix]:
                raise SystemExit(f"{path}: record domain does not match its stream prefix")
            source = require_manifest_dict(record.get("source"), f"{path} source")
            for key, expected in expected_sources[prefix].items():
                if source.get(key) != expected:
                    raise SystemExit(f"{path}: source {key} does not match the fixed corpus")
            if not isinstance(record.get("segments"), list) or not record["segments"]:
                raise SystemExit(f"{path}: each stream must retain source segments")
            segment_bytes = 0
            for segment in record["segments"]:
                segment = require_manifest_dict(segment, f"{path} segment")
                if set(segment) != {"source_id", "start_byte", "end_byte"}:
                    raise SystemExit(f"{path}: segment fields do not match the provenance contract")
                if not isinstance(segment.get("source_id"), str) or not segment["source_id"]:
                    raise SystemExit(f"{path}: segment source_id must be non-empty text")
                if prefix == "zhwiki":
                    article = segment["source_id"].removeprefix("article/")
                    if article == segment["source_id"] or not article.isdecimal():
                        raise SystemExit(f"{path}: invalid Chinese Wikipedia article id")
                start = segment.get("start_byte")
                end = segment.get("end_byte")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                ):
                    raise SystemExit(f"{path}: invalid source byte range")
                segment_bytes += end - start
            stream_id = record.get("stream_id")
            if stream_id not in stream_bytes or segment_bytes != stream_bytes[stream_id]:
                raise SystemExit(f"{path}: source segments do not cover the stream byte count")
        if ids != [f"{prefix}-{index:02d}" for index in range(STREAMS_PER_DOMAIN)]:
            raise SystemExit(f"{path}: stream order mismatch")

    quick_ids = set(expected_quick)
    if not quick_ids < set(expected_ids):
        raise SystemExit("quick mode must be a strict subset of full mode")
    print(
        f"[corpus] verified id={CORPUS_ID} streams={len(streams)} bytes={total_bytes}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "eval/corpora/perplexity-1m",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wikitext-snapshot", type=Path)
    parser.add_argument("--pg19-snapshot", type=Path)
    parser.add_argument("--pg19-assets", type=Path)
    parser.add_argument("--wikipedia-snapshot", type=Path)
    parser.add_argument("--ninfer-repo", type=Path)
    args = parser.parse_args()
    if args.tokenizer is not None:
        args.tokenizer = args.tokenizer.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.check:
        if args.tokenizer is None:
            parser.error("generation requires --tokenizer")
        required = (
            "wikitext_snapshot",
            "pg19_snapshot",
            "pg19_assets",
            "wikipedia_snapshot",
            "ninfer_repo",
        )
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error("generation requires " + ", ".join(missing))
        for name in required:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main() -> int:
    args = parse_args()
    if args.check:
        verify(args.output)
    else:
        tokenizer = load_tokenizer(args.tokenizer)
        build(args, tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
