"""Exercise the implemented NInfer HTTP product contract with the standard library."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# A one-pixel PNG. The target frontend performs its normal resize/patch expansion.
_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes


def request(base_url: str, method: str, path: str, payload: Any | None = None) -> Response:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return Response(
                status=response.status,
                content_type=response.headers.get_content_type(),
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ContractError(f"{method} {path} returned HTTP {error.code}: {detail}") from error


def json_response(
    base_url: str, method: str, path: str, payload: Any | None = None
) -> dict[str, Any]:
    response = request(base_url, method, path, payload)
    if response.status != 200 or response.content_type != "application/json":
        raise ContractError(
            f"{method} {path} returned status={response.status} content-type={response.content_type}"
        )
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise ContractError(f"{method} {path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{method} {path} did not return a JSON object")
    return value


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if json_response(base_url, "GET", "/health") == {"status": "ok"}:
                return
        except (ContractError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContractError(f"server did not become healthy within {timeout:g}s: {last_error}")


def require_usage(usage: Any, prompt_key: str, completion_key: str) -> tuple[int, int]:
    if not isinstance(usage, dict):
        raise ContractError("response usage is not an object")
    prompt = usage.get(prompt_key)
    completion = usage.get(completion_key)
    if not isinstance(prompt, int) or prompt <= 0:
        raise ContractError(f"invalid {prompt_key}: {prompt!r}")
    if not isinstance(completion, int) or completion < 0:
        raise ContractError(f"invalid {completion_key}: {completion!r}")
    return prompt, completion


def openai_nonstream(base_url: str, model: str, messages: list[dict[str, Any]], *, max_tokens: int,
                     stop: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    if stop is not None:
        payload["stop"] = stop
    response = json_response(base_url, "POST", "/v1/chat/completions", payload)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContractError("OpenAI response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ContractError("OpenAI response is missing assistant content")
    if choice.get("finish_reason") not in {"stop", "length", "tool_calls"}:
        raise ContractError(f"invalid OpenAI finish_reason: {choice.get('finish_reason')!r}")
    prompt, completion = require_usage(response.get("usage"), "prompt_tokens", "completion_tokens")
    usage = response["usage"]
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("OpenAI total_tokens does not equal prompt + completion")
    return response


def parse_openai_stream(response: Response) -> tuple[str, str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            f"OpenAI stream returned status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_role = False
    saw_done = False
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(lines) != 1:
            raise ContractError(f"malformed OpenAI SSE block: {block!r}")
        if lines[0] == "[DONE]":
            if saw_done:
                raise ContractError("OpenAI stream emitted [DONE] more than once")
            saw_done = True
            continue
        if saw_done:
            raise ContractError("OpenAI stream emitted data after [DONE]")
        try:
            event = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("OpenAI stream contained invalid JSON") from error
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise ContractError("OpenAI stream event is missing choices")
        event_usage = event.get("usage")
        if event_usage is not None:
            if choices:
                raise ContractError("OpenAI usage event unexpectedly contains choices")
            if usage is not None:
                raise ContractError("OpenAI stream emitted usage more than once")
            if finish_reason is None:
                raise ContractError("OpenAI stream emitted usage before its finish event")
            usage = event_usage
            continue
        if usage is not None:
            raise ContractError("OpenAI stream emitted an event after usage")
        if len(choices) != 1:
            raise ContractError("ordinary OpenAI stream event must contain one choice")
        if finish_reason is not None:
            raise ContractError("OpenAI stream emitted an event after its finish event")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ContractError("OpenAI stream choice is missing delta")
        if "role" in delta:
            if saw_role or delta.get("role") != "assistant":
                raise ContractError("invalid or duplicate assistant-role event")
            saw_role = True
        if "content" in delta:
            if not isinstance(delta["content"], str):
                raise ContractError("OpenAI content delta is not a string")
            content.append(delta["content"])
        if "reasoning_content" in delta:
            if not isinstance(delta["reasoning_content"], str):
                raise ContractError("OpenAI reasoning delta is not a string")
            reasoning.append(delta["reasoning_content"])
        reason = choice.get("finish_reason")
        if reason is not None:
            if finish_reason is not None or reason not in {"stop", "length", "tool_calls"}:
                raise ContractError(f"invalid or duplicate OpenAI finish reason: {reason!r}")
            finish_reason = reason
    if not saw_role or not saw_done or finish_reason is None or usage is None:
        raise ContractError("OpenAI stream did not complete its role/finish/usage/[DONE] contract")
    prompt, completion = require_usage(usage, "prompt_tokens", "completion_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("OpenAI streamed total_tokens does not equal prompt + completion")
    return "".join(content), "".join(reasoning), finish_reason, usage


def response_text(response: dict[str, Any]) -> tuple[str, str]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ContractError("Responses output is not an array")
    content: list[str] = []
    reasoning: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ContractError("Responses output contains a non-object Item")
        item_type = item.get("type")
        parts = item.get("content", [])
        if item_type in {"message", "reasoning"} and not isinstance(parts, list):
            raise ContractError(f"Responses {item_type} content is not an array")
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise ContractError("Responses text content part has the wrong shape")
            if part.get("type") == "output_text":
                content.append(part["text"])
            elif part.get("type") == "reasoning_text":
                reasoning.append(part["text"])
            else:
                raise ContractError(f"unexpected Responses content type: {part.get('type')!r}")
    return "".join(content), "".join(reasoning)


def require_responses_usage(usage: Any) -> tuple[int, int]:
    prompt, completion = require_usage(usage, "input_tokens", "output_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("Responses total_tokens does not equal input + output")
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict) or not isinstance(
        input_details.get("cached_tokens"), int
    ):
        raise ContractError("Responses cached-token usage is missing")
    if not isinstance(output_details, dict) or not isinstance(
        output_details.get("reasoning_tokens"), int
    ):
        raise ContractError("Responses reasoning-token usage is missing")
    return prompt, completion


def responses_nonstream(
    base_url: str,
    model: str,
    input_value: Any,
    *,
    store: bool,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": store,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    response = json_response(base_url, "POST", "/v1/responses", payload)
    if response.get("object") != "response" or response.get("status") not in {
        "completed",
        "incomplete",
    }:
        raise ContractError("Responses non-streaming envelope has the wrong shape")
    if not isinstance(response.get("id"), str) or not response["id"].startswith("resp_"):
        raise ContractError("Responses non-streaming envelope has an invalid id")
    if response.get("model") != model or response.get("store") is not store:
        raise ContractError("Responses non-streaming request fields were not echoed")
    require_responses_usage(response.get("usage"))
    response_text(response)
    return response


def parse_responses_stream(response: Response) -> tuple[str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            "Responses stream returned "
            f"status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    terminal: dict[str, Any] | None = None
    expected_sequence = 0
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        event_lines = [line[7:] for line in block.splitlines() if line.startswith("event: ")]
        data_lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(event_lines) != 1 or len(data_lines) != 1 or data_lines[0] == "[DONE]":
            raise ContractError(f"malformed Responses SSE block: {block!r}")
        try:
            event = json.loads(data_lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("Responses stream contained invalid JSON") from error
        event_type = event.get("type")
        if event_type != event_lines[0]:
            raise ContractError("Responses SSE event name differs from JSON type")
        if event.get("sequence_number") != expected_sequence:
            raise ContractError("Responses sequence_number is not contiguous")
        expected_sequence += 1
        if terminal is not None:
            raise ContractError("Responses stream emitted data after its terminal event")
        if event_type == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses output_text delta is not a string")
            content.append(event["delta"])
        elif event_type == "response.reasoning_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses reasoning delta is not a string")
            reasoning.append(event["delta"])
        elif event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            value = event.get("response")
            if not isinstance(value, dict):
                raise ContractError("Responses terminal event is missing its Response")
            terminal = value
    if terminal is None or terminal.get("status") not in {"completed", "incomplete"}:
        raise ContractError("Responses stream did not terminate successfully")
    terminal_content, terminal_reasoning = response_text(terminal)
    if "".join(content) != terminal_content or "".join(reasoning) != terminal_reasoning:
        raise ContractError("Responses deltas do not reconstruct terminal output Items")
    require_responses_usage(terminal.get("usage"))
    return terminal_content, terminal_reasoning, terminal


def exercise(base_url: str, model: str) -> dict[str, Any]:
    models = json_response(base_url, "GET", "/v1/models")
    entries = models.get("data")
    if models.get("object") != "list" or not isinstance(entries, list) or len(entries) != 1:
        raise ContractError("model-list response has the wrong shape")
    if entries[0].get("id") != model or entries[0].get("owned_by") != "ninfer":
        raise ContractError("model-list response does not identify the configured NInfer model")
    single_model = json_response(base_url, "GET", f"/v1/models/{model}")
    if single_model.get("id") != model:
        raise ContractError("single-model response has the wrong id")

    anthropic_prompt = {
        "model": model,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "Reply briefly."}],
    }
    counted = json_response(base_url, "POST", "/v1/messages/count_tokens", anthropic_prompt)
    input_tokens = counted.get("input_tokens")
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ContractError("count_tokens returned a non-positive input_tokens value")

    messages = [{"role": "user", "content": "Reply with a single short word."}]
    stops = ["__NINFER_SMOKE_UNLIKELY_STOP__"]
    nonstream = openai_nonstream(base_url, model, messages, max_tokens=4, stop=stops)
    expected_message = nonstream["choices"][0]["message"]
    expected_content = expected_message.get("content", "")
    expected_reasoning = expected_message.get("reasoning_content", "")
    if nonstream["usage"]["completion_tokens"] <= 0:
        raise ContractError("OpenAI request completed without producing a token")
    if not expected_content and not expected_reasoning:
        raise ContractError("OpenAI request completed without publishing output bytes")
    stream_payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 4,
        "temperature": 0,
        "stop": stops,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    streamed = request(base_url, "POST", "/v1/chat/completions", stream_payload)
    content, reasoning, stream_finish, stream_usage = parse_openai_stream(streamed)
    if content != expected_content:
        raise ContractError("streamed content differs from the non-streaming greedy response")
    if reasoning != expected_reasoning:
        raise ContractError("streamed reasoning differs from the non-streaming greedy response")
    if stream_finish != nonstream["choices"][0]["finish_reason"]:
        raise ContractError("streamed and non-streaming finish reasons differ")
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
    ):
        if stream_usage.get(key) != nonstream["usage"].get(key):
            raise ContractError(f"streamed and non-streaming usage differs for {key}")

    responses_input = "Reply with a single short word."
    response_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": responses_input},
    )
    if response_count.get("object") != "response.input_tokens" or not isinstance(
        response_count.get("input_tokens"), int
    ):
        raise ContractError("Responses input_tokens returned the wrong shape")
    response_sync = responses_nonstream(
        base_url, model, responses_input, store=False
    )
    response_sync_text, response_sync_reasoning = response_text(response_sync)
    response_prompt_tokens, response_output_tokens = require_responses_usage(
        response_sync.get("usage")
    )
    if response_prompt_tokens != response_count["input_tokens"]:
        raise ContractError("Responses input_tokens differs from generation usage")
    response_stream_payload = {
        "model": model,
        "input": responses_input,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": False,
        "stream": True,
    }
    response_stream = request(
        base_url, "POST", "/v1/responses", response_stream_payload
    )
    streamed_text, streamed_reasoning, response_stream_terminal = parse_responses_stream(
        response_stream
    )
    if (streamed_text, streamed_reasoning) != (
        response_sync_text,
        response_sync_reasoning,
    ):
        raise ContractError("Responses streaming output differs from greedy non-streaming output")
    _, streamed_output_tokens = require_responses_usage(response_stream_terminal.get("usage"))
    if streamed_output_tokens != response_output_tokens:
        raise ContractError("Responses streaming output-token usage differs")

    stored_response = responses_nonstream(
        base_url, model, "Remember the code word ORCHID. Reply briefly.", store=True
    )
    stored_id = stored_response.get("id")
    if not isinstance(stored_id, str) or not stored_id.startswith("resp_"):
        raise ContractError("stored Response has an invalid id")
    retrieved = json_response(base_url, "GET", f"/v1/responses/{stored_id}")
    if retrieved != stored_response:
        raise ContractError("retrieved Response differs from the created Response")
    input_items = json_response(
        base_url, "GET", f"/v1/responses/{stored_id}/input_items?order=asc"
    )
    if input_items.get("object") != "list" or len(input_items.get("data", [])) != 1:
        raise ContractError("Responses input_items list has the wrong shape")
    continuation_input = "What code word was given? Reply with only that word."
    continuation = responses_nonstream(
        base_url,
        model,
        continuation_input,
        store=True,
        previous_response_id=stored_id,
    )
    standalone_continuation_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": continuation_input},
    )["input_tokens"]
    continuation_prompt_tokens, _ = require_responses_usage(continuation.get("usage"))
    if continuation.get("previous_response_id") != stored_id or (
        continuation_prompt_tokens <= standalone_continuation_count
    ):
        raise ContractError("previous_response_id did not reconstruct stored context")
    continuation_id = continuation.get("id")
    for response_id in (continuation_id, stored_id):
        deleted = json_response(base_url, "DELETE", f"/v1/responses/{response_id}")
        if deleted != {"id": response_id, "object": "response.deleted", "deleted": True}:
            raise ContractError("Responses delete returned the wrong object")

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                {"type": "text", "text": "What is visible? Answer briefly."},
            ],
        }
    ]
    image_response = openai_nonstream(base_url, model, image_messages, max_tokens=2)
    image_prompt_tokens, _ = require_usage(
        image_response.get("usage"), "prompt_tokens", "completion_tokens"
    )
    if image_prompt_tokens <= input_tokens:
        raise ContractError("image request did not expand the prompt through the Vision frontend")

    anthropic = json_response(base_url, "POST", "/v1/messages", anthropic_prompt)
    if anthropic.get("type") != "message" or anthropic.get("role") != "assistant":
        raise ContractError("Anthropic response has the wrong envelope")
    if anthropic.get("stop_reason") not in {"end_turn", "max_tokens", "tool_use"}:
        raise ContractError(f"invalid Anthropic stop_reason: {anthropic.get('stop_reason')!r}")
    blocks = anthropic.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise ContractError("Anthropic response content is empty")
    anthropic_input_tokens, _ = require_usage(
        anthropic.get("usage"), "input_tokens", "output_tokens"
    )
    if anthropic_input_tokens != input_tokens:
        raise ContractError("Anthropic usage input_tokens differs from count_tokens")

    return {
        "format": "ninfer_serve_contract_v2",
        "model": model,
        "count_tokens": input_tokens,
        "openai_finish_reason": stream_finish,
        "openai_completion_tokens": stream_usage["completion_tokens"],
        "responses_input_tokens": response_prompt_tokens,
        "responses_output_tokens": response_output_tokens,
        "image_prompt_tokens": image_prompt_tokens,
        "anthropic_stop_reason": anthropic["stop_reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--health-timeout", type=float, default=300.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_for_health(base_url, args.health_timeout)
    print(json.dumps(exercise(base_url, args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
