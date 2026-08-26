"""Run the Qwen3.6 thinking-preservation fixture through a real ninfer-serve process."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class TestFailure(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(base_url: str, method: str, path: str, payload: Any | None = None) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise TestFailure(f"{method} {path} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise TestFailure(f"{method} {path} failed: {error}") from error
    if not isinstance(value, dict):
        raise TestFailure(f"{method} {path} did not return a JSON object")
    return value


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise TestFailure(f"ninfer-serve exited during startup with code {process.returncode}")
        try:
            if request_json(base_url, "GET", "/health") == {"status": "ok"}:
                return
        except TestFailure as error:
            last_error = error
        time.sleep(0.25)
    raise TestFailure(f"ninfer-serve was not healthy after {timeout:g}s: {last_error}")


def chat_payload(model: str, fixture: dict[str, Any], messages_key: str) -> dict[str, Any]:
    payload = dict(fixture["chat"]["request"])
    payload["model"] = model
    payload["messages"] = fixture["chat"][messages_key]
    return payload


def chat_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TestFailure("Chat Completions response does not contain one choice")
    return choices[0]


def chat_prompt_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int):
        raise TestFailure("Chat Completions response does not contain prompt token usage")
    return int(usage["prompt_tokens"])


def response_payload(model: str, input_text: str, preserve: bool | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": True,
    }
    if preserve is not None:
        payload["chat_template_kwargs"] = {"preserve_thinking": preserve}
    return payload


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
    return events


def protocol_events(events: list[dict[str, Any]], event: str, protocol: str) -> list[dict[str, Any]]:
    return [
        item
        for item in events
        if item.get("event") == event and item.get("request", {}).get("protocol") == protocol
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def exercise(base_url: str, fixture: dict[str, Any], log_path: Path, backend: str) -> dict[str, Any]:
    models = request_json(base_url, "GET", "/v1/models")
    entries = models.get("data")
    require(isinstance(entries, list) and len(entries) == 1, "server did not expose one model")
    model = entries[0].get("id")
    require(isinstance(model, str) and model, "server model id is invalid")

    chat_payloads = [
        chat_payload(model, fixture, "first_messages"),
        chat_payload(model, fixture, "first_tool_messages"),
        chat_payload(model, fixture, "second_tool_messages"),
        chat_payload(model, fixture, "reset_messages"),
        chat_payload(model, fixture, "second_tool_messages"),
    ]
    chat_responses = [
        request_json(base_url, "POST", "/v1/chat/completions", payload)
        for payload in chat_payloads
    ]
    restored_choice = chat_choice(chat_responses[2])
    cold_choice = chat_choice(chat_responses[4])
    require(
        restored_choice.get("message") == cold_choice.get("message")
        and restored_choice.get("finish_reason") == cold_choice.get("finish_reason"),
        "restored and full-reset requests produced different greedy results",
    )

    closed_messages = [
        *fixture["chat"]["second_tool_messages"],
        fixture["chat"]["next_user_message"],
    ]
    closed_payload = chat_payload(model, fixture, "second_tool_messages")
    closed_payload["messages"] = closed_messages
    closed_payload["chat_template_kwargs"] = {"preserve_thinking": False}
    closed_without_thinking = request_json(
        base_url, "POST", "/v1/chat/completions", closed_payload
    )
    preserved_payload = dict(closed_payload)
    preserved_payload["chat_template_kwargs"] = {"preserve_thinking": True}
    closed_with_thinking = request_json(
        base_url, "POST", "/v1/chat/completions", preserved_payload
    )
    stripped_prompt_tokens = chat_prompt_tokens(closed_without_thinking)
    preserved_prompt_tokens = chat_prompt_tokens(closed_with_thinking)
    require(
        preserved_prompt_tokens > stripped_prompt_tokens,
        "preserving closed-turn reasoning did not increase the rendered prompt",
    )

    responses_fixture = fixture["responses"]
    parent = request_json(
        base_url,
        "POST",
        "/v1/responses",
        response_payload(model, responses_fixture["parent_input"], True),
    )
    parent_id = parent.get("id")
    require(isinstance(parent_id, str) and parent_id, "Responses parent id is invalid")

    inherited_payload = response_payload(
        model, responses_fixture["inherited_child_input"], None
    )
    inherited_payload["previous_response_id"] = parent_id
    request_json(base_url, "POST", "/v1/responses", inherited_payload)

    changed_payload = response_payload(model, responses_fixture["changed_child_input"], False)
    changed_payload["previous_response_id"] = parent_id
    request_json(base_url, "POST", "/v1/responses", changed_payload)

    events = read_events(log_path)
    chat_done = protocol_events(events, "request_done", "openai_chat_completions")
    require(len(chat_done) == 7, f"expected 7 Chat request_done events, found {len(chat_done)}")
    paths = [item.get("result", {}).get("prefix_reuse_path") for item in chat_done]
    require(
        paths == [
            "root",
            "private_turn_closure",
            "private_turn_closure",
            "root",
            "root",
            "private_turn_closure",
            "root",
        ],
        f"unexpected Chat reuse paths: {paths}",
    )
    first_restore = chat_done[1]["result"].get("prefix_cache_hit_tokens")
    second_restore = chat_done[2]["result"].get("prefix_cache_hit_tokens")
    require(
        isinstance(first_restore, int)
        and first_restore > 0
        and second_restore == first_restore,
        "tool-loop requests did not restore the same turn checkpoint",
    )
    for index in (1, 2):
        speculative = chat_done[index].get("speculative", {})
        require(speculative.get("backend") == backend, "request used the wrong speculative backend")
        require(speculative.get("rounds", 0) > 0, "request did not execute ReplaySSM rounds")

    chat_start = protocol_events(events, "request_start", "openai_chat_completions")
    require(
        len(chat_start) == 7
        and all(
            item.get("request", {}).get("preserve_thinking") is False
            for item in chat_start[:6]
        )
        and chat_start[6].get("request", {}).get("preserve_thinking") is True,
        "Chat requests did not resolve preserve_thinking consistently",
    )

    responses_start = protocol_events(events, "request_start", "openai_responses")
    require(len(responses_start) == 3, "expected three Responses request_start events")
    response_semantics = [
        (
            item["request"].get("preserve_thinking"),
            item["request"].get("preserve_thinking_semantic_change"),
        )
        for item in responses_start
    ]
    require(
        response_semantics == [(True, False), (True, False), (False, True)],
        f"unexpected Responses preserve semantics: {response_semantics}",
    )
    responses_done = protocol_events(events, "request_done", "openai_responses")
    require(len(responses_done) == 3, "expected three Responses request_done events")
    response_paths = [
        item.get("result", {}).get("prefix_reuse_path") for item in responses_done
    ]
    require(
        response_paths
        == ["root", "private_response_replay", "root"],
        f"unexpected Responses reuse paths: {response_paths}",
    )

    return {
        "backend": backend,
        "model": model,
        "turn_checkpoint_frontier": first_restore,
        "chat_reuse_paths": paths,
        "closed_turn_prompt_tokens": {
            "stripped": stripped_prompt_tokens,
            "preserved": preserved_prompt_tokens,
        },
        "responses_preserve_semantics": response_semantics,
        "responses_reuse_paths": response_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--backend", choices=("mtp", "dflash"), required=True)
    parser.add_argument("--server-bin", type=Path, default=Path("build/apps/ninfer-serve"))
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/serve/qwen3_6_thinking_preservation.json"),
    )
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    server_bin = args.server_bin.resolve()
    fixture_path = args.fixture.resolve()
    if not artifact.is_file():
        raise SystemExit(f"artifact does not exist: {artifact}")
    if not server_bin.is_file():
        raise SystemExit(f"server binary does not exist: {server_bin}")
    with fixture_path.open("r", encoding="utf-8") as source:
        fixture = json.load(source)
    if fixture.get("schema_version") != 1:
        raise SystemExit("unsupported fixture schema")

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix=f"ninfer-{args.backend}-thinking-") as temporary:
        temporary_path = Path(temporary)
        request_log = temporary_path / "requests.jsonl"
        server_log = temporary_path / "server.log"
        command = [
            str(server_bin),
            str(artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-context",
            "1024",
            "--kv-capacity",
            "1024",
            "--prefill-chunk",
            "128",
            "--max-concurrency",
            "1",
            "--max-pending-requests",
            "4",
            "--log-stats-interval-ms",
            "0",
            "--request-log-jsonl",
            str(request_log),
            "--spec",
            args.backend,
            "--draft-tokens",
            "3",
            "--lm-head-draft",
            "--greedy",
        ]
        with server_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_server(base_url, process, args.startup_timeout)
                result = exercise(base_url, fixture, request_log, args.backend)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as error:
                output.flush()
                tail = server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                detail = "\n".join(tail)
                raise SystemExit(f"{error}\n\nlast server log lines:\n{detail}") from error
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)


if __name__ == "__main__":
    main()
