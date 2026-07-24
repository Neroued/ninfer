#!/usr/bin/env python3
"""Run the fixed two-target serving corpus performance evaluation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import http.client
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "examples/cli/manifest.json"

TARGET_MODEL_IDS = {
    "qwen3_6_35b_a3b": "qwen3.6-35b-a3b",
    "qwen3_6_27b": "qwen3.6-27b",
}
TARGET_ORDER = tuple(TARGET_MODEL_IDS)

SEEDS = (
    7632647173703958409,
    7968175640111700217,
    912910298659544128,
    9060622443728853932,
    4939353812939007330,
)

NIAH_FIXTURES = (
    "long_niah_8k",
    "long_niah_64k",
    "long_niah_128k",
    "long_niah_256k",
)

LONG_DECODE_FIXTURES = (
    "long_decode_aime26_01",
    "long_decode_aime26_15",
    "long_decode_aime26_30",
)

SCENARIO_FIXTURES = {
    "code": (
        "scenario_code_cuda",
        "scenario_code_python",
        "scenario_code_typescript",
    ),
    "story": (
        "scenario_story_zh_scifi",
        "scenario_story_en_mystery",
        "scenario_story_zh_dialogue",
    ),
    "translation": (
        "scenario_translation_zh_en",
        "scenario_translation_en_zh",
        "scenario_translation_markdown",
    ),
    "structured": (
        "scenario_structured_jsonl",
        "scenario_structured_csv",
        "scenario_structured_sql",
    ),
}

WARMUP_FIXTURE = "text_smoke_zh"
RUN_ARTIFACT_TYPE = "ninfer_serve_corpus_result"
RUN_SCHEMA_VERSION = 3
SERVER_LOG_ARTIFACT_TYPE = "ninfer_serve_request_log"
SERVER_LOG_SCHEMA_VERSION = 2
STARTUP_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
LOG_EVENT_TIMEOUT_SECONDS = 10.0


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    messages: list[dict[str, Any]]
    thinking: bool
    max_new: int
    suite: str
    category: str | None = None


@dataclasses.dataclass(frozen=True)
class RunSpec:
    target: str
    model_id: str
    artifact: Path
    mtp_draft_tokens: int
    fixture: Fixture
    seed: int

    @property
    def mtp_mode(self) -> str:
        return f"mtp{self.mtp_draft_tokens}"

    @property
    def key(self) -> tuple[str, int, str, int]:
        return (self.target, self.mtp_draft_tokens, self.fixture.name, self.seed)


class CampaignError(RuntimeError):
    pass


class ServerLogTail:
    def __init__(
        self, path: Path, process: subprocess.Popen[bytes], initial_offset: int
    ) -> None:
        self.path = path
        self.process = process
        self.offset = initial_offset
        self.buffer = b""
        self.pending: list[dict[str, Any]] = []

    def _check_process(self) -> None:
        returncode = self.process.poll()
        if returncode is not None:
            raise CampaignError(f"ninfer-serve exited unexpectedly with status {returncode}")

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return
        self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        for raw_line in lines:
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignError(f"invalid serving JSONL record in {self.path}: {exc}") from exc
            if not isinstance(event, dict):
                raise CampaignError(f"serving JSONL record is not an object in {self.path}")
            self.pending.append(event)

    def wait_for(
        self,
        predicate: Any,
        description: str,
        timeout: float = LOG_EVENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            self._read_new()
            for index, event in enumerate(self.pending):
                if predicate(event):
                    return self.pending.pop(index)
            self._check_process()
            if time.monotonic() >= deadline:
                raise CampaignError(f"timed out waiting for {description} in {self.path}")
            time.sleep(0.05)


class RunningServer:
    def __init__(
        self,
        command: Sequence[str],
        host: str,
        port: int,
        log_path: Path,
    ) -> None:
        self.command = list(command)
        self.host = host
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.tail: ServerLogTail | None = None

    def __enter__(self) -> "RunningServer":
        initial_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        self.process = subprocess.Popen(self.command, cwd=REPO_ROOT)
        self.tail = ServerLogTail(self.log_path, self.process, initial_offset)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def wait_until_ready(self) -> dict[str, Any]:
        if self.process is None or self.tail is None:
            raise CampaignError("server process was not started")
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            returncode = self.process.poll()
            if returncode is not None:
                raise CampaignError(f"ninfer-serve exited during startup with status {returncode}")
            connection = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
            try:
                connection.request("GET", "/health", headers={"Connection": "close"})
                response = connection.getresponse()
                body = response.read()
                if response.status == 200:
                    try:
                        health = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        health = None
                    if health == {"status": "ok"}:
                        break
            except OSError:
                pass
            finally:
                connection.close()
            if time.monotonic() >= deadline:
                raise CampaignError(
                    f"timed out waiting for ninfer-serve at http://{self.host}:{self.port}"
                )
            time.sleep(0.2)

        return self.tail.wait_for(
            lambda event: event.get("event") == "server_start",
            "server_start event",
        )

    def wait_for_request_done(self, server_instance_id: str) -> dict[str, Any]:
        if self.tail is None:
            raise CampaignError("server log tail is unavailable")

        def matches(event: dict[str, Any]) -> bool:
            if event.get("server_instance_id") != server_instance_id:
                return False
            if event.get("event") == "request_error":
                message = event.get("error", {}).get("message", "unknown generation error")
                raise CampaignError(f"serving request failed: {message}")
            return event.get("event") == "request_done"

        return self.tail.wait_for(matches, "request_done event")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve",
        type=Path,
        default=REPO_ROOT / "build/apps/ninfer-serve",
        help="ninfer-serve executable",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="TARGET=PATH",
        help="artifact for a registered target; pass once for each of the two targets",
    )
    parser.add_argument("--output", type=Path, required=True, help="campaign output directory")
    parser.add_argument("--port", type=int, default=8080, help="loopback serving port")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser.parse_args(argv)


def parse_artifacts(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: dict[str, Path] = {}
    for value in values:
        target, separator, raw_path = value.partition("=")
        if not separator or not target or not raw_path:
            raise CampaignError(f"invalid --artifact value {value!r}; expected TARGET=PATH")
        if target not in TARGET_MODEL_IDS:
            expected = ", ".join(TARGET_MODEL_IDS)
            raise CampaignError(f"unsupported artifact target {target!r}; expected {expected}")
        if target in parsed:
            raise CampaignError(f"duplicate artifact target: {target}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise CampaignError(f"artifact not found: {path}")
        parsed[target] = path
    missing = set(TARGET_MODEL_IDS) - set(parsed)
    if missing:
        raise CampaignError(f"missing artifact target(s): {', '.join(sorted(missing))}")
    return [(target, parsed[target]) for target in TARGET_ORDER]


def fixture_metadata(name: str) -> tuple[str, str | None]:
    if name in NIAH_FIXTURES:
        return "long_niah", None
    if name in LONG_DECODE_FIXTURES:
        return "long_decode", "reasoning"
    for category, names in SCENARIO_FIXTURES.items():
        if name in names:
            return "scenario", category
    if name == WARMUP_FIXTURE:
        return "warmup", None
    raise CampaignError(f"fixture has no campaign assignment: {name}")


def load_fixtures() -> dict[str, Fixture]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"failed to read {MANIFEST_PATH}: {exc}") from exc
    cases = {case["name"]: case for case in manifest["cases"]}
    selected_names = (
        *NIAH_FIXTURES,
        *LONG_DECODE_FIXTURES,
        *(name for names in SCENARIO_FIXTURES.values() for name in names),
        WARMUP_FIXTURE,
    )
    fixtures: dict[str, Fixture] = {}
    for name in selected_names:
        try:
            case = cases[name]
            messages_path = MANIFEST_PATH.parent / case["messages"]
            messages = json.loads(messages_path.read_text(encoding="utf-8"))
            suite, category = fixture_metadata(name)
            fixtures[name] = Fixture(
                name=name,
                messages=messages,
                thinking=bool(case["thinking"]),
                max_new=int(case["max_new"]),
                suite=suite,
                category=category,
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CampaignError(f"failed to load fixture {name!r} from the examples manifest: {exc}") from exc
    return fixtures


def block_fixture_names(mtp_draft_tokens: int) -> tuple[str, ...]:
    scenarios = tuple(name for names in SCENARIO_FIXTURES.values() for name in names)
    if mtp_draft_tokens == 0:
        return NIAH_FIXTURES
    if mtp_draft_tokens == 3:
        return (*LONG_DECODE_FIXTURES, *scenarios)
    raise CampaignError(f"unsupported MTP draft window: {mtp_draft_tokens}")


def build_specs(
    artifacts: Sequence[tuple[str, Path]], fixtures: dict[str, Fixture]
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for target, artifact in artifacts:
        for mtp_draft_tokens in (0, 3):
            for fixture_name in block_fixture_names(mtp_draft_tokens):
                for seed in SEEDS:
                    specs.append(
                        RunSpec(
                            target=target,
                            model_id=TARGET_MODEL_IDS[target],
                            artifact=artifact,
                            mtp_draft_tokens=mtp_draft_tokens,
                            fixture=fixtures[fixture_name],
                            seed=seed,
                        )
                    )
    return specs


def request_payload(model_id: str, fixture: Fixture, seed: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": fixture.messages,
        "max_completion_tokens": fixture.max_new,
        "seed": seed,
        "stream": False,
        "enable_thinking": fixture.thinking,
    }


def post_json(connection: http.client.HTTPConnection, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Connection": "keep-alive",
            },
        )
        response = connection.getresponse()
        response_body = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise CampaignError(f"HTTP request failed: {exc}") from exc
    if response.status != 200:
        detail = response_body.decode("utf-8", errors="replace")
        raise CampaignError(f"HTTP {response.status} {response.reason}: {detail}")
    try:
        parsed = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"serving response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CampaignError("serving response is not a JSON object")
    return parsed


def require_server_log_identity(event: dict[str, Any], event_name: str) -> None:
    identity = (
        event.get("artifact_type"),
        event.get("schema_version"),
        event.get("event"),
    )
    expected = (SERVER_LOG_ARTIFACT_TYPE, SERVER_LOG_SCHEMA_VERSION, event_name)
    if identity != expected:
        raise CampaignError(f"unexpected serving log identity {identity!r}; expected {expected!r}")


def validate_server_start(event: dict[str, Any], spec: RunSpec, device: int) -> str:
    require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    actual = {
        "device": engine.get("device"),
        "max_context": engine.get("max_context"),
        "prefill_chunk": engine.get("prefill_chunk"),
        "kv_cache": engine.get("kv_cache"),
        "cuda_graph": engine.get("cuda_graph"),
        "prefix_reuse": engine.get("prefix_reuse"),
        "speculative_backend": engine.get("speculative_backend"),
        "speculative_draft_window": engine.get("speculative_draft_window"),
        "proposal_head": engine.get("proposal_head"),
    }
    expected = {
        "device": device,
        "max_context": 262144,
        "prefill_chunk": 1024,
        "kv_cache": "int8-group64",
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": "mtp" if spec.mtp_draft_tokens else "none",
        "speculative_draft_window": spec.mtp_draft_tokens,
        "proposal_head": "optimized" if spec.mtp_draft_tokens else "full",
    }
    if actual != expected:
        raise CampaignError(f"server_start Engine configuration mismatch: {actual!r}")
    if event.get("artifact", {}).get("target") != spec.target:
        raise CampaignError(
            "loaded artifact target mismatch: "
            f"{event.get('artifact', {}).get('target')!r} != {spec.target!r}"
        )
    if event.get("server", {}).get("public_model_id") != spec.model_id:
        raise CampaignError("server_start public model id does not match the campaign target")
    server_instance_id = event.get("server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise CampaignError("server_start has no server_instance_id")
    return server_instance_id


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def build_result_record(
    spec: RunSpec,
    payload: dict[str, Any],
    response: dict[str, Any],
    server_event: dict[str, Any],
) -> dict[str, Any]:
    require_server_log_identity(server_event, "request_done")
    request = server_event.get("request", {})
    result = server_event.get("result", {})
    timings = server_event.get("timings_seconds", {})
    speculative = server_event.get("speculative", {})

    expected_request = {
        "model": spec.model_id,
        "requested_output_tokens": spec.fixture.max_new,
        "enable_thinking": spec.fixture.thinking,
        "seed": spec.seed,
    }
    actual_request = {
        "model": request.get("model"),
        "requested_output_tokens": request.get("requested_output_tokens"),
        "enable_thinking": request.get("enable_thinking"),
        "seed": request.get("sampling", {}).get("seed"),
    }
    if actual_request != expected_request:
        raise CampaignError(
            f"request_done does not match the submitted request: {actual_request!r}"
        )

    try:
        prompt_tokens = int(result["prompt_tokens"])
        completion_tokens = int(result["completion_tokens"])
        prepare_seconds = float(timings["prepare"])
        vision_seconds = float(timings["vision"])
        prefill_seconds = float(timings["prefill"])
        decode_seconds = float(timings["decode"])
        total_seconds = float(timings["total"])
        backend = str(speculative["backend"])
        mtp_rounds = int(speculative["rounds"])
        drafted_tokens = int(speculative["drafted_tokens"])
        accepted_tokens = int(speculative["accepted_tokens"])
        fallback_steps = int(speculative["fallback_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"request_done is missing required metrics: {exc}") from exc

    expected_backend = "mtp" if spec.mtp_draft_tokens else "none"
    if backend != expected_backend:
        raise CampaignError(
            f"request_done speculative backend {backend!r} != {expected_backend!r}"
        )

    usage = response.get("usage", {})
    if (
        usage.get("prompt_tokens") != prompt_tokens
        or usage.get("completion_tokens") != completion_tokens
    ):
        raise CampaignError("HTTP response usage does not match request_done token counts")

    decode_tokens = max(completion_tokens - 1, 0)
    metrics = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "finish_reason": result.get("finish_reason"),
        "prepare_seconds": prepare_seconds,
        "vision_seconds": vision_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
        "prefill_tok_s": safe_ratio(float(prompt_tokens), prefill_seconds),
        "server_ttft_ms": 1000.0 * (prepare_seconds + vision_seconds + prefill_seconds),
        "decode_tok_s": safe_ratio(float(decode_tokens), decode_seconds),
        "mtp_rounds": mtp_rounds,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "mtp_acceptance": safe_ratio(float(accepted_tokens), float(drafted_tokens)),
        "mtp_tokens_per_round": (
            1.0 + accepted_tokens / mtp_rounds if mtp_rounds > 0 else None
        ),
        "fallback_steps": fallback_steps,
    }
    return {
        "artifact_type": RUN_ARTIFACT_TYPE,
        "schema_version": RUN_SCHEMA_VERSION,
        "target": spec.target,
        "model": spec.model_id,
        "artifact_path": str(spec.artifact),
        "fixture": spec.fixture.name,
        "suite": spec.fixture.suite,
        "category": spec.fixture.category,
        "seed": spec.seed,
        "mtp_mode": spec.mtp_mode,
        "mtp_draft_tokens": spec.mtp_draft_tokens,
        "request": payload,
        "response": response,
        "server_event": server_event,
        "metrics": metrics,
    }


def record_key(record: dict[str, Any]) -> tuple[str, int, str, int]:
    try:
        return (
            str(record["target"]),
            int(record["mtp_draft_tokens"]),
            str(record["fixture"]),
            int(record["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"invalid corpus result record key: {exc}") from exc


def load_existing_records(
    path: Path,
    expected_specs: dict[tuple[str, int, str, int], RunSpec],
) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    records: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = (record.get("artifact_type"), record.get("schema_version"))
                expected_identity = (RUN_ARTIFACT_TYPE, RUN_SCHEMA_VERSION)
                if identity != expected_identity:
                    raise CampaignError(
                        f"{path}:{line_number}: unexpected record identity {identity!r}"
                    )
                key = record_key(record)
                if key not in expected_specs:
                    raise CampaignError(f"{path}:{line_number}: result is outside this campaign")
                if key in records:
                    raise CampaignError(f"{path}:{line_number}: duplicate result for {key!r}")
                spec = expected_specs[key]
                if Path(record.get("artifact_path", "")).resolve() != spec.artifact:
                    raise CampaignError(
                        f"{path}:{line_number}: artifact path differs from the current command"
                    )
                records[key] = record
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"failed to read existing results from {path}: {exc}") from exc
    return records


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def server_command(
    serve: Path,
    spec: RunSpec,
    server_log: Path,
    port: int,
    device: int,
) -> list[str]:
    command = [
        str(serve),
        str(spec.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        spec.model_id,
        "--max-context",
        "262144",
        "--prefill-chunk",
        "1024",
        "--device",
        str(device),
        "--request-log-jsonl",
        str(server_log),
        "--kv-dtype",
        "int8",
        "--no-prefix-reuse",
    ]
    if spec.mtp_draft_tokens:
        command.extend(
            [
                "--spec",
                "mtp",
                "--draft-tokens",
                str(spec.mtp_draft_tokens),
                "--lm-head-draft",
            ]
        )
    return command


def run_block(
    serve: Path,
    block_specs: Sequence[RunSpec],
    fixtures: dict[str, Fixture],
    output_dir: Path,
    port: int,
    device: int,
    run_handle: Any,
    records: dict[tuple[str, int, str, int], dict[str, Any]],
    completed_before_block: int,
    total: int,
) -> None:
    first = block_specs[0]
    server_log = output_dir / "server" / f"{first.target}_{first.mtp_mode}.jsonl"
    command = server_command(serve, first, server_log, port, device)
    print(
        f"start {first.target}/{first.mtp_mode}: {len(block_specs)} missing request(s)",
        flush=True,
    )
    with RunningServer(command, "127.0.0.1", port, server_log) as server:
        server_start = server.wait_until_ready()
        server_instance_id = validate_server_start(server_start, first, device)

        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS
        )
        last_request_id: int | None = None
        try:
            warmup = fixtures[WARMUP_FIXTURE]
            post_json(connection, request_payload(first.model_id, warmup, SEEDS[0]))
            warmup_done = server.wait_for_request_done(server_instance_id)
            require_server_log_identity(warmup_done, "request_done")
            last_request_id = int(warmup_done.get("request", {}).get("request_id"))

            for block_index, spec in enumerate(block_specs, start=1):
                payload = request_payload(spec.model_id, spec.fixture, spec.seed)
                response = post_json(connection, payload)
                request_done = server.wait_for_request_done(server_instance_id)
                request_id = int(request_done.get("request", {}).get("request_id"))
                if request_id != last_request_id + 1:
                    raise CampaignError(
                        f"non-sequential serving request id {request_id}; expected {last_request_id + 1}"
                    )
                last_request_id = request_id
                record = build_result_record(spec, payload, response, request_done)
                append_record(run_handle, record)
                records[spec.key] = record
                completed = completed_before_block + block_index
                print(
                    f"[{completed}/{total}] {spec.target}/{spec.mtp_mode} "
                    f"{spec.fixture.name} seed={spec.seed}",
                    flush=True,
                )
        finally:
            connection.close()


def metric_values(records: Iterable[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record["metrics"].get(name)
        if value is not None:
            values.append(float(value))
    return values


def sample_stats(records: Sequence[dict[str, Any]], name: str) -> tuple[int, float | None, float | None]:
    values = metric_values(records, name)
    if not values:
        return 0, None, None
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return len(values), mean, stddev


def select_records(
    records: dict[tuple[str, int, str, int], dict[str, Any]],
    target: str,
    mtp_draft_tokens: int,
    fixtures: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        records[(target, mtp_draft_tokens, fixture, seed)]
        for fixture in fixtures
        for seed in SEEDS
    ]


SUMMARY_FIELDS = (
    "section",
    "target",
    "group",
    "fixture",
    "mtp_mode",
    "samples",
    "prompt_tokens_mean",
    "prompt_tokens_stddev",
    "prefill_tok_s_mean",
    "prefill_tok_s_stddev",
    "server_ttft_ms_mean",
    "server_ttft_ms_stddev",
    "completion_tokens_mean",
    "completion_tokens_stddev",
    "decode_tok_s_mean",
    "decode_tok_s_stddev",
    "mtp_acceptance_mean",
    "mtp_acceptance_stddev",
    "mtp_tokens_per_round_mean",
    "mtp_tokens_per_round_stddev",
)


def set_stats(
    row: dict[str, Any],
    prefix: str,
    records: Sequence[dict[str, Any]],
    metric: str,
) -> None:
    _, mean, stddev = sample_stats(records, metric)
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_stddev"] = stddev


def summary_row(
    section: str,
    target: str,
    group: str,
    fixture: str,
    mtp_draft_tokens: int,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "section": section,
        "target": target,
        "group": group,
        "fixture": fixture,
        "mtp_mode": f"mtp{mtp_draft_tokens}",
        "samples": len(records),
    }
    set_stats(row, "prompt_tokens", records, "prompt_tokens")
    set_stats(row, "prefill_tok_s", records, "prefill_tok_s")
    set_stats(row, "server_ttft_ms", records, "server_ttft_ms")
    set_stats(row, "completion_tokens", records, "completion_tokens")
    set_stats(row, "decode_tok_s", records, "decode_tok_s")
    set_stats(row, "mtp_acceptance", records, "mtp_acceptance")
    set_stats(row, "mtp_tokens_per_round", records, "mtp_tokens_per_round")
    return row


def build_summary_rows(
    records: dict[tuple[str, int, str, int], dict[str, Any]],
    target_order: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_order:
        for fixture in NIAH_FIXTURES:
            rows.append(
                summary_row(
                    "context_profile",
                    target,
                    fixture,
                    fixture,
                    0,
                    select_records(records, target, 0, (fixture,)),
                )
            )

        for fixture in LONG_DECODE_FIXTURES:
            rows.append(
                summary_row(
                    "long_decode",
                    target,
                    "reasoning",
                    fixture,
                    3,
                    select_records(records, target, 3, (fixture,)),
                )
            )

        for category, fixture_names in SCENARIO_FIXTURES.items():
            for fixture in fixture_names:
                rows.append(
                    summary_row(
                        "scenario_fixture",
                        target,
                        category,
                        fixture,
                        3,
                        select_records(records, target, 3, (fixture,)),
                    )
                )

            rows.append(
                summary_row(
                    "scenario_category",
                    target,
                    category,
                    "",
                    3,
                    select_records(records, target, 3, fixture_names),
                )
            )
    return rows


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise CampaignError("summary contains a non-finite value")
    return value


def format_mean_stddev(row: dict[str, Any], prefix: str, digits: int = 1) -> str:
    mean = row.get(f"{prefix}_mean")
    stddev = row.get(f"{prefix}_stddev")
    if mean is None or stddev is None:
        return "—"
    return f"{float(mean):.{digits}f} ± {float(stddev):.{digits}f}"


def format_percent_mean_stddev(row: dict[str, Any], prefix: str) -> str:
    mean = row.get(f"{prefix}_mean")
    stddev = row.get(f"{prefix}_stddev")
    if mean is None or stddev is None:
        return "—"
    return f"{100.0 * float(mean):.1f}% ± {100.0 * float(stddev):.1f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def write_summaries(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row.get(field)) for field in SUMMARY_FIELDS} for row in rows
        )

    context_rows = [row for row in rows if row["section"] == "context_profile"]
    long_decode_rows = [row for row in rows if row["section"] == "long_decode"]
    category_rows = [row for row in rows if row["section"] == "scenario_category"]

    context_table = markdown_table(
        (
            "Target",
            "Fixture",
            "n",
            "Prompt tokens",
            "Prefill tok/s",
            "Server TTFT ms",
            "Decode tok/s",
        ),
        [
            (
                row["target"],
                row["fixture"],
                str(row["samples"]),
                format_mean_stddev(row, "prompt_tokens"),
                format_mean_stddev(row, "prefill_tok_s"),
                format_mean_stddev(row, "server_ttft_ms"),
                format_mean_stddev(row, "decode_tok_s"),
            )
            for row in context_rows
        ],
    )
    long_decode_table = markdown_table(
        (
            "Target",
            "Fixture",
            "n",
            "Completion tokens",
            "Decode tok/s",
            "MTP acceptance",
            "MTP tokens/round",
        ),
        [
            (
                row["target"],
                row["fixture"],
                str(row["samples"]),
                format_mean_stddev(row, "completion_tokens"),
                format_mean_stddev(row, "decode_tok_s"),
                format_percent_mean_stddev(row, "mtp_acceptance"),
                format_mean_stddev(row, "mtp_tokens_per_round", digits=2),
            )
            for row in long_decode_rows
        ],
    )
    scenario_table = markdown_table(
        (
            "Target",
            "Category",
            "n",
            "Decode tok/s",
            "MTP acceptance",
            "MTP tokens/round",
        ),
        [
            (
                row["target"],
                row["group"],
                str(row["samples"]),
                format_mean_stddev(row, "decode_tok_s"),
                format_percent_mean_stddev(row, "mtp_acceptance"),
                format_mean_stddev(row, "mtp_tokens_per_round", digits=2),
            )
            for row in category_rows
        ],
    )

    markdown = (
        "# Serving corpus performance summary\n\n"
        "All values are arithmetic mean ± sample standard deviation.\n\n"
        "## MTP0 context-length profile\n\n"
        f"{context_table}\n\n"
        "## MTP3 long-decode reasoning\n\n"
        f"{long_decode_table}\n\n"
        "## MTP3 cross-scenario decode\n\n"
        f"{scenario_table}\n"
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.port < 1 or args.port > 65535:
        raise CampaignError("--port must be in [1, 65535]")
    if args.device < 0:
        raise CampaignError("--device must be nonnegative")

    serve = args.serve.expanduser().resolve()
    if not serve.is_file():
        raise CampaignError(f"ninfer-serve executable not found: {serve}")
    if not os.access(serve, os.X_OK):
        raise CampaignError(f"ninfer-serve is not executable: {serve}")

    artifacts = parse_artifacts(args.artifact)
    fixtures = load_fixtures()
    specs = build_specs(artifacts, fixtures)
    expected_specs = {spec.key: spec for spec in specs}
    if len(expected_specs) != 190:
        raise CampaignError(f"internal campaign size is {len(expected_specs)}, expected 190")

    output_dir = args.output.expanduser().resolve()
    (output_dir / "server").mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.jsonl"
    records = load_existing_records(run_path, expected_specs)
    print(f"resume state: {len(records)}/190 formal request(s) complete", flush=True)

    with run_path.open("a", encoding="utf-8") as run_handle:
        for target, _ in artifacts:
            for mtp_draft_tokens in (0, 3):
                block_specs = [
                    spec
                    for spec in specs
                    if spec.target == target
                    and spec.mtp_draft_tokens == mtp_draft_tokens
                    and spec.key not in records
                ]
                if not block_specs:
                    print(f"skip {target}/mtp{mtp_draft_tokens}: block complete", flush=True)
                    continue
                run_block(
                    serve,
                    block_specs,
                    fixtures,
                    output_dir,
                    args.port,
                    args.device,
                    run_handle,
                    records,
                    len(records),
                    len(expected_specs),
                )

    missing = set(expected_specs) - set(records)
    if missing:
        raise CampaignError(f"campaign ended with {len(missing)} missing formal request(s)")
    target_order = [target for target, _ in artifacts]
    summary_rows = build_summary_rows(records, target_order)
    write_summaries(summary_rows, output_dir)
    print(f"completed 190 formal requests; summary: {output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted; completed results remain in run.jsonl", file=sys.stderr)
        raise SystemExit(130) from None
