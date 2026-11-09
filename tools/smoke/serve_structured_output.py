"""End-to-end structured-output check against a running ninfer-serve (Chat, Responses, Anthropic).

For every protocol, streaming and not, it sends constrained requests (JSON object, JSON Schema,
and for Chat the structured_outputs extension: choice, regex, grammar) and checks the reply with
an independent validator (`jsonschema`, Draft 2020-12; Python `re` for regex). It then checks that
unsupported combinations fail with HTTP 400 on the right field (`param` and JSON Pointer) before
generation, and that an unconstrained request still works. The robustness cases (post-audit) check
that a thinking budget cutting a multi-byte character keeps the reply valid JSON with a single
close, that JSON whitespace runs are bounded, and that failing constrained requests do not affect
free requests running next to them (`--robustness only` runs just these).
The hostile inputs of the audit (deep
schema, huge choice, slow compilation, empty languages, ambiguous grammars) must fail alone with a
400 or a constraint_budget_exceeded error, leave the server up, and not slow an unconstrained
request running next to them (--no-hostile skips them).

Usage (from the repository root, with `pip install -r tests/text/requirements.txt`):

    python -m tools.smoke.serve_structured_output --base-url http://127.0.0.1:18080 --model qwen

Exit status 0 when every check passes. A report is printed and, with --json, written to a file.
It refuses the production ports (8081, 8099) unless --allow-port is given for that port.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

try:
    import jsonschema
except ImportError:  # pragma: no cover - reported at startup
    jsonschema = None

PRODUCTION_PORTS = {8081, 8099}

SCHEMAS: dict[str, dict[str, Any]] = {
    "flat": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 40},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age", "active"],
        "additionalProperties": False,
    },
    "nested_enum": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "weather": {
                "type": "object",
                "properties": {
                    "condition": {"enum": ["sunny", "cloudy", "rain", "snow"]},
                    "celsius": {"type": "number", "minimum": -60, "maximum": 60},
                },
                "required": ["condition", "celsius"],
                "additionalProperties": False,
            },
        },
        "required": ["city", "weather"],
        "additionalProperties": False,
    },
    "array": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, "score": {"type": "integer"}},
                    "required": ["label", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

PROMPTS = {
    "flat": "Invent a person and describe them as JSON with name, age and active.",
    "nested_enum": "Give the current weather for Rome as JSON.",
    "array": "List three fruits with a score from 1 to 10, as JSON.",
    "json_object": "Return a JSON object describing a cat with two fields.",
    "choice": "Is the sky blue on a clear day? Answer yes or no.",
    "regex": "Give a three-digit number.",
    "grammar": "Answer with a colour: red or green.",
}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HttpReply:
    status: int
    body: bytes


def post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    raw: bytes | None = None,
) -> HttpReply:
    data = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpReply(response.status, response.read())
    except urllib.error.HTTPError as error:
        return HttpReply(error.code, error.read())


def sse_events(body: bytes) -> list[tuple[str | None, Any]]:
    events: list[tuple[str | None, Any]] = []
    for block in body.decode().split("\n\n"):
        name = None
        data = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if not data:
            continue
        raw = "\n".join(data)
        events.append((name, raw if raw == "[DONE]" else json.loads(raw)))
    return events


# --- protocol adapters: build a request, extract (content, finish) from the reply -----------------


def chat_payload(model: str, prompt: str, extra: dict[str, Any], stream: bool, thinking: bool, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": thinking},
        **extra,
    }
    return "/v1/chat/completions", payload


def chat_extract(reply: HttpReply, stream: bool) -> tuple[str, str, dict[str, Any]]:
    if not stream:
        body = json.loads(reply.body)
        choice = body["choices"][0]
        return choice["message"].get("content") or "", choice["finish_reason"], {}
    content, finish = [], ""
    for _, event in sse_events(reply.body):
        if event == "[DONE]" or not event.get("choices"):
            continue
        choice = event["choices"][0]
        content.append(choice.get("delta", {}).get("content") or "")
        finish = choice.get("finish_reason") or finish
    return "".join(content), finish, {}


def responses_payload(model: str, prompt: str, extra: dict[str, Any], stream: bool, thinking: bool, max_tokens: int):
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": stream,
        "store": False,
        "reasoning": {"effort": "medium" if thinking else "none"},
        **extra,
    }
    return "/v1/responses", payload


def responses_extract(reply: HttpReply, stream: bool) -> tuple[str, str, dict[str, Any]]:
    if not stream:
        body = json.loads(reply.body)
        echo = body.get("text", {}).get("format")
        text = "".join(
            part.get("text", "")
            for item in body.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        return text, body.get("status", ""), {"echo": echo}
    text, status, echo_created = [], "", None
    for name, event in sse_events(reply.body):
        if name == "response.created":
            echo_created = event["response"].get("text", {}).get("format")
        elif name == "response.output_text.delta":
            text.append(event["delta"])
        elif name in ("response.completed", "response.incomplete", "response.failed"):
            status = event["response"].get("status", name)
    return "".join(text), status, {"echo": echo_created}


def anthropic_payload(model: str, prompt: str, extra: dict[str, Any], stream: bool, thinking: bool, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "thinking": {"type": "adaptive"} if thinking else {"type": "disabled"},
        **extra,
    }
    return "/v1/messages", payload


def anthropic_extract(reply: HttpReply, stream: bool) -> tuple[str, str, dict[str, Any]]:
    if not stream:
        body = json.loads(reply.body)
        text = "".join(block.get("text", "") for block in body["content"] if block["type"] == "text")
        return text, body.get("stop_reason") or "", {}
    text, stop = [], ""
    for name, event in sse_events(reply.body):
        if name == "content_block_delta" and event["delta"].get("type") == "text_delta":
            text.append(event["delta"]["text"])
        elif name == "message_delta":
            stop = event["delta"].get("stop_reason") or stop
    return "".join(text), stop, {}


PROTOCOLS: dict[str, tuple[Callable, Callable, set[str], dict[str, str]]] = {
    "chat": (chat_payload, chat_extract, {"stop"}, {}),
    "responses": (responses_payload, responses_extract, {"completed"}, {}),
    "anthropic": (anthropic_payload, anthropic_extract, {"end_turn"}, {"anthropic-version": "2023-06-01"}),
}


def format_field(protocol: str, kind: str, schema: dict[str, Any] | None, name: str = "answer") -> dict[str, Any]:
    if protocol == "chat":
        if kind == "json_object":
            return {"response_format": {"type": "json_object"}}
        return {
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        }
    if protocol == "responses":
        if kind == "json_object":
            return {"text": {"format": {"type": "json_object"}}}
        return {"text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}}
    return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}


# --- checks --------------------------------------------------------------------------------------


def check_valid(
    args: argparse.Namespace, protocol: str, case: str, extra: dict[str, Any], validate: Callable[[str], str | None]
) -> list[Result]:
    build, extract, finished, headers = PROTOCOLS[protocol]
    results = []
    for stream in (False, True):
        for thinking in args.thinking:
            label = f"{protocol}/{case}/{'stream' if stream else 'once'}/{'think' if thinking else 'no-think'}"
            for attempt in range(args.repeat):
                path, payload = build(args.model, PROMPTS[case], extra, stream, thinking, args.max_tokens)
                started = time.monotonic()
                reply = post(args.base_url, path, payload, headers, args.timeout)
                seconds = time.monotonic() - started
                name = label if args.repeat == 1 else f"{label}#{attempt}"
                if reply.status != 200:
                    results.append(Result(name, False, f"HTTP {reply.status}: {reply.body[:300]!r}", seconds))
                    continue
                try:
                    content, finish, info = extract(reply, stream)
                except (ValueError, KeyError, IndexError) as error:
                    results.append(Result(name, False, f"unreadable reply ({error}): {reply.body[:200]!r}", seconds))
                    continue
                if finish not in finished:
                    # max_tokens still applies: a truncated reply is reported, not validated.
                    results.append(Result(name, False, f"not finished ({finish}); content {content[:120]!r}", seconds))
                    continue
                problem = validate(content)
                if protocol == "responses" and "text" in extra:
                    if info.get("echo") != extra["text"]["format"]:
                        problem = problem or f"text.format echo differs: {info.get('echo')!r}"
                results.append(Result(name, problem is None, problem or content[:160], seconds))
    return results


def check_unconstrained(args: argparse.Namespace, protocol: str) -> list[Result]:
    """A request without a format must keep working as before (both stream modes)."""
    build, extract, finished, headers = PROTOCOLS[protocol]
    results = []
    for stream in (False, True):
        path, payload = build(args.model, "Say hello in five words.", {}, stream, False, 256)
        started = time.monotonic()
        reply = post(args.base_url, path, payload, headers, args.timeout)
        seconds = time.monotonic() - started
        name = f"{protocol}/unconstrained/{'stream' if stream else 'once'}"
        if reply.status != 200:
            results.append(Result(name, False, f"HTTP {reply.status}: {reply.body[:300]!r}", seconds))
            continue
        try:
            content, finish, _ = extract(reply, stream)
        except (ValueError, KeyError, IndexError) as error:
            results.append(Result(name, False, f"unreadable reply ({error}): {reply.body[:200]!r}", seconds))
            continue
        results.append(Result(name, finish in finished and bool(content), f"{finish}: {content[:100]!r}", seconds))
    return results


def json_validator(schema: dict[str, Any] | None) -> Callable[[str], str | None]:
    def validate(content: str) -> str | None:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as error:
            return f"not JSON ({error}): {content[:160]!r}"
        if schema is None:
            return None if isinstance(value, dict) else f"not an object: {content[:160]!r}"
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(value), key=lambda e: e.path)
        return None if not errors else f"schema violation: {errors[0].message}"

    return validate


def check_errors(args: argparse.Namespace) -> list[Result]:
    """Unsupported or invalid requests: 400 on the field, before any generation."""
    schema = SCHEMAS["flat"]
    bad_keyword = {"type": "object", "properties": {"when": {"type": "string", "format": "date"}}}
    cases: list[tuple[str, str, dict[str, Any], str, str | None]] = [
        # (name, protocol, extra fields, expected param (prefix match with *), expected code)
        (
            "chat/unsupported_keyword",
            "chat",
            format_field("chat", "json_schema", bad_keyword),
            "response_format.json_schema.schema/properties/when/format",
            "unsupported_json_schema",
        ),
        (
            "chat/stop_combination",
            "chat",
            {**format_field("chat", "json_object", None), "stop": ["}"]},
            "response_format",
            "unsupported_output_combination",
        ),
        (
            "chat/tools_active",
            "chat",
            {
                **format_field("chat", "json_schema", schema),
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            },
            "response_format.json_schema.schema",
            "unsupported_output_combination",
        ),
        (
            "chat/two_constraints",
            "chat",
            {**format_field("chat", "json_object", None), "structured_outputs": {"choice": ["a"]}},
            "structured_outputs.choice",
            "unsupported_output_combination",
        ),
        ("chat/bad_regex", "chat", {"structured_outputs": {"regex": "("}}, "structured_outputs.regex", "invalid_regex"),
        (
            "chat/bad_grammar",
            "chat",
            {"structured_outputs": {"grammar": "root ::= "}},
            "structured_outputs.grammar",
            "invalid_grammar",
        ),
        (
            "chat/missing_name",
            "chat",
            {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}},
            "response_format.json_schema.name",
            None,
        ),
        (
            "chat/required_tool_choice",
            "chat",
            {
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "tool_choice": "required",
            },
            "tool_choice",
            "tool_choice_not_supported",
        ),
        (
            "responses/unsupported_keyword",
            "responses",
            format_field("responses", "json_schema", bad_keyword),
            "text.format.schema/properties/when/format",
            "unsupported_json_schema",
        ),
        (
            "responses/inexact_number",
            "responses",
            format_field("responses", "json_schema", {"type": "number", "maximum": 0.30000000000000001}),
            "text.format.schema/maximum",
            "unsupported_json_schema",
        ),
        (
            "anthropic/unsupported_keyword",
            "anthropic",
            format_field("anthropic", "json_schema", bad_keyword),
            "output_config.format.schema/properties/when/format",
            None,
        ),
        (
            "anthropic/json_object",
            "anthropic",
            {"output_config": {"format": {"type": "json_object"}}},
            "output_config.format.type",
            None,
        ),
    ]
    results = []
    for name, protocol, extra, param, code in cases:
        build, _, _, headers = PROTOCOLS[protocol]
        path, payload = build(args.model, "hello", extra, False, False, 16)
        raw = None
        if name == "responses/inexact_number":
            # json.dumps would shorten the literal (and re-parsing it would too): send the exact
            # spelling the server must reject as raw bytes.
            raw = json.dumps(payload).replace("0.3", "0.30000000000000001").encode()
        started = time.monotonic()
        reply = post(args.base_url, path, payload, headers, args.timeout, raw)
        seconds = time.monotonic() - started
        try:
            body = json.loads(reply.body)
        except json.JSONDecodeError:
            body = {}
        error = body.get("error") or {}
        if protocol == "anthropic":
            # No param in the Anthropic error object: the location leads the message.
            got_param = error.get("message", "").split(": ", 1)[0]
            ok = reply.status == 400 and error.get("type") == "invalid_request_error" and got_param == param
        else:
            got_param = error.get("param")
            ok = reply.status == 400 and got_param == param and (code is None or error.get("code") == code)
        results.append(
            Result(f"error/{name}", ok, f"HTTP {reply.status} param={got_param!r} code={error.get('code')!r}", seconds)
        )
    return results


# --- robustness (post-audit): cases that once stopped the engine or broke the JSON ---------------

BUDGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["answer", "count"],
    "additionalProperties": False,
}

# Reasoning made of emoji: the thinking budget stops the model between the byte-level tokens of one
# character (audit B1: the forced control suffix was refused and the engine stopped for everyone;
# B4: a second close reached the content).
EMOJI_PROMPT = (
    "Count the flamingos. Here is the list: "
    + "🦩🦚" * 300
    + "\nIn your thinking, go through the list emoji by emoji, writing each emoji you see "
    "followed by nothing else, before answering. Answer with a JSON object with the fields answer "
    "(one sentence) and count (the number of flamingos)."
)

# Audit I3: a model stuck on blanks ran to max_tokens; whitespace runs are now bounded.
WHITESPACE_LIMIT = 64
WHITESPACE_PROMPT = (
    "Output 300 newline characters, then 300 spaces, then 300 more newlines, and only then a "
    "JSON object with one field a equal to 1."
)


def get_status(base_url: str, path: str, timeout: float = 30.0) -> int:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return -1


def budget_request(args: argparse.Namespace, budget: int) -> Result:
    payload = {
        "model": args.model,
        "max_tokens": budget + 400,
        "temperature": 0,
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "messages": [{"role": "user", "content": EMOJI_PROMPT}],
        "output_config": {"format": {"type": "json_schema", "schema": BUDGET_SCHEMA}},
    }
    started = time.monotonic()
    reply = post(args.base_url, "/v1/messages", payload, {"anthropic-version": "2023-06-01"}, args.timeout)
    seconds = time.monotonic() - started
    name = f"robust/budget_cut/{budget}"
    if reply.status != 200:
        return Result(name, False, f"HTTP {reply.status}: {reply.body[:300]!r}", seconds)
    body = json.loads(reply.body)
    text = "".join(block.get("text", "") for block in body["content"] if block["type"] == "text")
    thinking = "".join(block.get("thinking", "") for block in body["content"] if block["type"] == "thinking")
    marker = thinking.find("Considering the limited time")
    cut = marker > 0 and "�" in thinking[max(0, marker - 8) : marker]
    problem = json_validator(BUDGET_SCHEMA)(text)
    if "</think>" in text:
        problem = f"second close in the content: {text[:120]!r}"
    if body.get("stop_reason") != "end_turn":
        problem = problem or f"stop_reason {body.get('stop_reason')!r}"
    detail = (problem or text[:100]) + (" [cut character at the budget]" if cut else "")
    return Result(name, problem is None, detail, seconds, {"cut": cut})


def check_robustness(args: argparse.Namespace) -> list[Result]:
    import threading

    results: list[Result] = []
    # 1. Thinking budget cutting characters, JSON Schema (the audit's failing budgets, both models),
    #    submitted together.
    budgets = [1024, 1025, 1026, 1027]
    slots: list[Result | None] = [None] * len(budgets)

    def run_budget(index: int) -> None:
        try:
            slots[index] = budget_request(args, budgets[index])
        except Exception as error:  # noqa: BLE001 - reported as a failure
            slots[index] = Result(f"robust/budget_cut/{budgets[index]}", False, repr(error))

    threads = [threading.Thread(target=run_budget, args=(i,)) for i in range(len(budgets))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results += [r for r in slots if r is not None]
    health = get_status(args.base_url, "/health")
    results.append(Result("robust/budget_cut/health_after", health == 200, f"/health {health}"))

    # 2. Whitespace runs are bounded: the reply must still end on its own, as one JSON object.
    path, payload = chat_payload(
        args.model, WHITESPACE_PROMPT, {"response_format": {"type": "json_object"}}, False, False, 1024
    )
    started = time.monotonic()
    reply = post(args.base_url, path, payload, {}, args.timeout)
    seconds = time.monotonic() - started
    if reply.status != 200:
        results.append(Result("robust/whitespace", False, f"HTTP {reply.status}: {reply.body[:200]!r}", seconds))
    else:
        content, finish, _ = chat_extract(reply, False)
        leading = len(content) - len(content.lstrip(" \t\r\n"))
        trailing = len(content) - len(content.rstrip(" \t\r\n"))
        problem = json_validator(None)(content)
        if finish != "stop":
            problem = problem or f"finish {finish!r}"
        if leading > WHITESPACE_LIMIT or trailing > WHITESPACE_LIMIT:
            problem = problem or f"whitespace run {leading}/{trailing} beyond {WHITESPACE_LIMIT}"
        detail = problem or f"leading {leading}: {content.strip()[:80]!r}"
        results.append(Result("robust/whitespace", problem is None, detail, seconds))

    # 3. Isolation: constrained requests that fail (empty language at the first token and after a
    #    prefix) and the budget cut, next to three free requests. The free ones must finish
    #    normally and the server must stay healthy.
    free_out: list[Any] = [None] * 3
    bad_out: list[Any] = [None] * 2
    cut_out: list[Result | None] = [None]
    bad_grammars = ["root ::= [^\\x00-\\U0010FFFF]", 'root ::= "ok" [^\\x00-\\U0010FFFF]']

    def run_free(index: int) -> None:
        prompts = ["Write a short paragraph about Rome.", "Explain how a fridge works.", "Tell a short story."]
        path, payload = chat_payload(args.model, prompts[index], {}, False, index == 2, 300)
        reply = post(args.base_url, path, payload, {}, args.timeout)
        free_out[index] = (reply.status, chat_extract(reply, False) if reply.status == 200 else reply.body[:200])

    def run_bad(index: int) -> None:
        path, payload = chat_payload(
            args.model,
            "Give me a short answer.",
            {"structured_outputs": {"grammar": bad_grammars[index]}},
            False,
            False,
            64,
        )
        reply = post(args.base_url, path, payload, {}, args.timeout)
        try:
            code = json.loads(reply.body).get("error", {}).get("code")
        except (ValueError, AttributeError):
            code = None
        bad_out[index] = (reply.status, code)

    def run_cut() -> None:
        try:
            cut_out[0] = budget_request(args, 1024)
        except Exception as error:  # noqa: BLE001
            cut_out[0] = Result("robust/isolation/budget_cut", False, repr(error))

    started = time.monotonic()
    threads = [threading.Thread(target=run_free, args=(i,)) for i in range(3)]
    threads += [threading.Thread(target=run_bad, args=(i,)) for i in range(2)]
    threads.append(threading.Thread(target=run_cut))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seconds = time.monotonic() - started
    for index, out in enumerate(free_out):
        if out is None:
            results.append(Result(f"robust/isolation/free{index}", False, "no reply", seconds))
            continue
        status, data = out
        ok = status == 200 and data[1] in ("stop", "length") and bool(data[0])
        detail = f"HTTP {status} finish {data[1]!r}" if status == 200 else f"HTTP {status}: {data!r}"
        results.append(Result(f"robust/isolation/free{index}", ok, detail, seconds))
    for index, out in enumerate(bad_out):
        status, code = out if out is not None else (None, None)
        ok = status == 400 and code in ("constraint_dead_end", "invalid_grammar")
        results.append(Result(f"robust/isolation/failing{index}", ok, f"HTTP {status} code={code}", seconds))
    cut = cut_out[0] or Result("robust/isolation/budget_cut", False, "no result")
    results.append(Result("robust/isolation/budget_cut", cut.ok, cut.detail, cut.seconds))
    health = get_status(args.base_url, "/health")
    path, payload = chat_payload(args.model, "Say hello in five words.", {}, False, False, 64)
    after = post(args.base_url, path, payload, {}, args.timeout)
    detail = f"/health {health}, next request HTTP {after.status}"
    results.append(Result("robust/isolation/after", health == 200 and after.status == 200, detail))
    return results


# --- hostile inputs (structured output limits, docs/serving.md) ----------------------------------


def nested_items_schema(levels: int) -> str:
    """JSON text of a schema nested `levels` deep; built as text (json.dumps would recurse)."""
    return '{"type":"array","items":' * levels + '{"type":"integer"}' + "}" * levels


def chat_raw(model: str, prompt: str, format_text: str, max_tokens: int, stream: bool = False) -> bytes:
    head = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return (
        head[:-1]
        + ', "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": '
        + format_text
        + "}}}"
    ).encode()


def healthy(args: argparse.Namespace) -> bool:
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def error_of(reply: HttpReply) -> dict[str, Any]:
    try:
        return json.loads(reply.body).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return {}


def check_hostile_rejections(args: argparse.Namespace) -> list[Result]:
    """Audit inputs refused with a 400 before generation; the server stays up after each."""
    optional = {f"p{i}": {"type": "string"} for i in range(1000)}
    cases: list[tuple[str, bytes | None, dict[str, Any], str, str, str]] = [
        # (name, raw body or None, extra fields, expected param prefix, code, message fragment)
        (
            # Below --max-request-json-depth (1000), past --constraint-max-depth (256): the schema is
            # refused at its JSON Pointer. Deeper, the whole body is refused before parsing.
            "deep_schema_300",
            chat_raw(args.model, "hi", nested_items_schema(300), 16),
            {},
            "response_format.json_schema.schema/items/items",
            "constraint_limit_exceeded",
            "--constraint-max-depth",
        ),
        (
            "deep_schema_5000",
            chat_raw(args.model, "hi", nested_items_schema(5000), 16),
            {},
            "",
            "request_json_too_deep",
            "--max-request-json-depth",
        ),
        (
            "deep_schema_100000",
            chat_raw(args.model, "hi", nested_items_schema(100000), 16),
            {},
            "",
            "request_json_too_deep",
            "--max-request-json-depth",
        ),
        (
            "choice_100000",
            None,
            {"structured_outputs": {"choice": [f"value_{i}_xyz" for i in range(100000)]}},
            "structured_outputs.choice",
            "constraint_limit_exceeded",
            "--constraint-max-items",
        ),
        (
            "optional_props_1000",
            None,
            format_field(
                "chat", "json_schema", {"type": "object", "properties": optional, "additionalProperties": False}
            ),
            "response_format.json_schema.schema",
            "constraint_limit_exceeded",
            "--constraint-compile-timeout-ms",
        ),
        (
            "gbnf_deep_1000",
            None,
            {"structured_outputs": {"grammar": "root ::= " + "(" * 1000 + '"a"' + ")" * 1000}},
            "structured_outputs.grammar",
            "constraint_limit_exceeded",
            "--constraint-max-depth",
        ),
        (
            "gbnf_empty_language",
            None,
            {"structured_outputs": {"grammar": "root ::= [^\\x00-\\U0010FFFF]"}},
            "structured_outputs.grammar",
            "invalid_grammar",
            "accepts no output",
        ),
        (
            "regex_empty_language",
            None,
            {"structured_outputs": {"regex": "[^\\s\\S]"}},
            "structured_outputs.regex",
            "invalid_regex",
            "accepts no output",
        ),
    ]
    results = []
    for name, raw, extra, param, code, fragment in cases:
        path, payload = chat_payload(args.model, "hi", extra, False, False, 16)
        started = time.monotonic()
        reply = post(args.base_url, path, payload, {}, args.timeout, raw)
        seconds = time.monotonic() - started
        error = error_of(reply)
        ok = (
            reply.status == 400
            and (error.get("param") or "").startswith(param)
            and error.get("code") == code
            and fragment in (error.get("message") or "")
        )
        alive = healthy(args)
        results.append(
            Result(
                f"hostile/{name}",
                ok and alive,
                f"HTTP {reply.status} code={error.get('code')!r} param={(error.get('param') or '')[:60]!r} "
                f"alive={alive} {(error.get('message') or '')[:80]!r}",
                seconds,
            )
        )
    return results


AMBIGUOUS = [
    # (name, structured_outputs) from the audit (B3): Earley work grows with the output.
    ("gbnf_ambiguous", {"grammar": 'root ::= e\ne ::= e e | "a" | "b"'}),
    ("regex_nested_star", {"regex": "(a*)*"}),
    ("regex_nested_alternation", {"regex": "((a|b)*)*"}),
]
# The same languages as plain patterns: a model that ends its answer before the matcher gets
# expensive has a valid, finished reply, which is fine too.
AMBIGUOUS_LANGUAGE = {"gbnf_ambiguous": "[ab]+", "regex_nested_star": "a*", "regex_nested_alternation": "[ab]*"}


def ambiguous_outcome(name: str, reply: HttpReply, stream: bool) -> tuple[bool, dict[str, Any]]:
    """True when the request failed alone with constraint_budget_exceeded, or finished in its
    language before reaching the budget."""
    if stream:
        events = [event for _, event in sse_events(reply.body) if isinstance(event, dict)]
        error = next((e.get("error") for e in events if e.get("error")), None) or {}
    else:
        error = error_of(reply) if reply.status != 200 else {}
    if error:
        return error.get("code") == "constraint_budget_exceeded", error
    if reply.status != 200:
        return False, {}
    content, finish, _ = chat_extract(reply, stream)
    finished = finish == "stop" and re.fullmatch(AMBIGUOUS_LANGUAGE[name], content) is not None
    return finished, {"message": f"finished in the language before the budget ({len(content)} chars)"}


AMBIGUOUS_PROMPT = "Write the letter a two thousand times, without spaces."


def chat_speed(args: argparse.Namespace, max_tokens: int) -> tuple[float, int, float]:
    """One unconstrained request: (decode tokens/s from the server timings, tokens, wall seconds)."""
    path, payload = chat_payload(
        args.model, "Write a long story about a lighthouse keeper.", {"temperature": 0}, False, False, max_tokens
    )
    started = time.monotonic()
    reply = post(args.base_url, path, payload, {}, args.timeout)
    seconds = time.monotonic() - started
    if reply.status != 200:
        return 0.0, 0, seconds
    body = json.loads(reply.body)
    tokens = body.get("usage", {}).get("completion_tokens", 0)
    return float(body.get("timings", {}).get("predicted_per_second") or 0.0), tokens, seconds


def check_hostile_generation(args: argparse.Namespace) -> list[Result]:
    """Ambiguous grammars fail alone (constraint_budget_exceeded) and quickly; an unconstrained
    request running next to them keeps its speed."""
    import threading

    results = []
    for name, outputs in AMBIGUOUS:
        for stream in (False, True):
            path, payload = chat_payload(
                args.model, AMBIGUOUS_PROMPT, {"structured_outputs": outputs}, stream, False, 3000
            )
            started = time.monotonic()
            reply = post(args.base_url, path, payload, {}, args.timeout)
            seconds = time.monotonic() - started
            ok, error = ambiguous_outcome(name, reply, stream)
            if error.get("code"):
                ok = ok and error.get("param") == f"structured_outputs.{next(iter(outputs))}"
            results.append(
                Result(
                    f"hostile/{name}/{'stream' if stream else 'once'}",
                    ok and healthy(args),
                    f"HTTP {reply.status} code={error.get('code')!r} {(error.get('message') or '')[:90]!r}",
                    seconds,
                )
            )

    # Speed of an unconstrained request alone, then next to every ambiguous request at once.
    alone, tokens_alone, wall_alone = chat_speed(args, args.speed_tokens)
    outcome: dict[str, HttpReply] = {}

    def hostile(name: str, outputs: dict[str, Any]) -> None:
        path, payload = chat_payload(args.model, AMBIGUOUS_PROMPT, {"structured_outputs": outputs}, False, False, 3000)
        outcome[name] = post(args.base_url, path, payload, {}, args.timeout)

    threads = [threading.Thread(target=hostile, args=case) for case in AMBIGUOUS]
    for thread in threads:
        thread.start()
    beside, tokens_beside, wall_beside = chat_speed(args, args.speed_tokens)
    for thread in threads:
        thread.join()
    failed_alone = all(ambiguous_outcome(name, reply, False)[0] for name, reply in outcome.items())
    ratio = beside / alone if alone else 0.0
    results.append(
        Result(
            "hostile/concurrent_speed",
            alone > 0 and ratio >= args.speed_ratio and failed_alone and healthy(args),
            f"alone {alone:.1f} tok/s ({tokens_alone} tok, {wall_alone:.1f}s), beside {len(AMBIGUOUS)} ambiguous "
            f"{beside:.1f} tok/s ({tokens_beside} tok, {wall_beside:.1f}s), ratio {ratio:.2f}; "
            f"ambiguous closed alone: {failed_alone}",
            wall_beside,
            {"alone_tps": alone, "beside_tps": beside, "ratio": ratio},
        )
    )
    return results


def check_deep_request(args: argparse.Namespace, protocols: list[str]) -> list[Result]:
    """Unconstrained requests whose JSON nests past --max-request-json-depth (a tool schema 100 000
    deep used to kill the server): 400 request_json_too_deep on every protocol, server up."""
    deep = '{"type":"array","items":' * 100000 + '{"type":"integer"}' + "}" * 100000
    tools = {
        "chat": '"tools": [{"type": "function", "function": {"name": "f", "parameters": ' + deep + "}}]",
        "responses": '"tools": [{"type": "function", "name": "f", "parameters": ' + deep + "}]",
        "anthropic": '"tools": [{"name": "f", "input_schema": ' + deep + "}]",
    }
    results = []
    for protocol in protocols:
        build, _, _, headers = PROTOCOLS[protocol]
        path, payload = build(args.model, "hi", {}, False, False, 16)
        raw = (json.dumps(payload)[:-1] + ", " + tools[protocol] + "}").encode()
        started = time.monotonic()
        reply = post(args.base_url, path, payload, headers, args.timeout, raw)
        seconds = time.monotonic() - started
        error = error_of(reply)
        code_ok = error.get("code") == "request_json_too_deep" or (
            protocol == "anthropic" and "--max-request-json-depth" in (error.get("message") or "")
        )
        alive = healthy(args)
        results.append(
            Result(
                f"deep_request/{protocol}",
                reply.status == 400 and code_ok and alive,
                f"HTTP {reply.status} code={error.get('code')!r} alive={alive} {(error.get('message') or '')[:80]!r}",
                seconds,
            )
        )
    return results


def diamond_schema(op: str, levels: int, leaf: dict[str, Any], salt: str) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for i in range(levels):
        nxt = f"#/$defs/d{i + 1}"
        defs[f"d{i}"] = {op: [{"$ref": nxt}, {"$ref": nxt}]} if i + 1 < levels else leaf
    defs["salt"] = {"title": salt}
    return {"$defs": defs, "$ref": "#/$defs/d0"}


OPEN_SCHEMA_PROMPTS = [
    "Invent a person. Give name, age, email, city and job as JSON.",
    "Describe a fictional customer with name, age, phone number and favourite product, as JSON.",
    "Create a user profile as JSON including name, age, hobbies and address.",
    "Return JSON for a book character: name, age, and anything else you find relevant.",
    "Give the name and age of a famous scientist as JSON.",
]


def check_open_schema(args: argparse.Namespace) -> list[Result]:
    """Third audit N9: an object with declared properties and no additionalProperties (the most
    common hand-written schema). The model wants to add keys; in strict mode (default, as vLLM) it
    cannot, so the reply is the declared object and no request fails on the matcher budget."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    results = []
    for i, prompt in enumerate(OPEN_SCHEMA_PROMPTS):
        path, payload = chat_payload(args.model, prompt, format_field("chat", "json_schema", schema), False, False, 300)
        started = time.monotonic()
        reply = post(args.base_url, path, payload, {}, args.timeout)
        seconds = time.monotonic() - started
        problem = None
        if reply.status != 200:
            problem = f"HTTP {reply.status} code={error_of(reply).get('code')!r}"
        else:
            content, finish, _ = chat_extract(reply, False)
            try:
                value = json.loads(content)
                if finish != "stop" or set(value) != {"name", "age"}:
                    problem = f"finish={finish} keys={sorted(value)}"
            except (json.JSONDecodeError, TypeError):
                problem = f"not JSON: {content[:80]!r}"
        results.append(Result(f"open_schema/{i}", problem is None, problem or "declared keys only", seconds))
    return results


def check_compilation_hostile(args: argparse.Namespace) -> list[Result]:
    """Second audit N1: allOf/$ref diamonds compiled for ever and blocked every later compilation.
    A diamond of identical references now compiles at once; an exponential validation (enum
    against an anyOf diamond) fails alone at the compilation deadline; meanwhile a small new schema
    and an unconstrained request are served."""
    import threading

    salt = str(time.time_ns())
    results: list[Result] = []
    outcome: dict[str, tuple[HttpReply, float]] = {}

    def send(name: str, extra: dict[str, Any], max_tokens: int = 64) -> None:
        path, payload = chat_payload(args.model, "Reply with a small JSON object.", extra, False, False, max_tokens)
        started = time.monotonic()
        outcome[name] = (post(args.base_url, path, payload, {}, args.timeout), time.monotonic() - started)

    slow = []
    for i in range(2):
        schema = diamond_schema("anyOf", 40, {"type": "integer"}, f"{salt}-{i}")
        del schema["$ref"]
        schema["allOf"] = [{"enum": [1]}, {"$ref": "#/$defs/d0"}]
        slow.append(schema)
    cases = [
        (
            "allof_diamond_0",
            format_field("chat", "json_schema", diamond_schema("allOf", 40, {"type": "object"}, salt + "a")),
        ),
        (
            "allof_diamond_1",
            format_field("chat", "json_schema", diamond_schema("allOf", 40, {"type": "object"}, salt + "b")),
        ),
        ("exponential_0", format_field("chat", "json_schema", slow[0])),
        ("exponential_1", format_field("chat", "json_schema", slow[1])),
    ]
    threads = [threading.Thread(target=send, args=case) for case in cases]
    for thread in threads:
        thread.start()
    time.sleep(0.5)
    small = {
        "type": "object",
        "properties": {f"k{salt}": {"type": "integer"}},
        "required": [f"k{salt}"],
        "additionalProperties": False,
    }
    send("small_schema_beside", format_field("chat", "json_schema", small))
    send("unconstrained_beside", {}, 16)
    for thread in threads:
        thread.join()
    for name, (reply, seconds) in outcome.items():
        error = error_of(reply) if reply.status != 200 else {}
        if name.startswith("exponential"):
            ok = (
                reply.status == 400
                and error.get("code") == "constraint_limit_exceeded"
                and "--constraint-compile-timeout-ms" in (error.get("message") or "")
                and seconds < 10
            )
        elif name.startswith("allof_diamond"):
            try:
                content, finish, _ = chat_extract(reply, False)
                ok = (
                    reply.status == 200 and isinstance(json.loads(content), dict)
                    if finish == "stop"
                    else reply.status == 200
                )
            except (ValueError, KeyError, IndexError):
                ok = False
        else:
            ok = reply.status == 200 and seconds < 10
        results.append(
            Result(
                f"compile_hostile/{name}",
                ok and healthy(args),
                f"HTTP {reply.status} code={error.get('code')!r} {(error.get('message') or '')[:90]!r}",
                seconds,
            )
        )
    return results


def _spin(stop: Any) -> None:
    while not stop.is_set():
        pass


def check_cpu_load(args: argparse.Namespace) -> list[Result]:
    """Second audit N3: the matcher budget is CPU time, so a legitimate expensive pattern on a
    saturated host must not fail, while an ambiguous grammar still does."""
    import multiprocessing

    stop = multiprocessing.Event()
    hogs = [multiprocessing.Process(target=_spin, args=(stop,), daemon=True) for _ in range(args.cpu_hogs)]
    for hog in hogs:
        hog.start()
    results: list[Result] = []
    try:
        domains = {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^([a-z0-9]{1,64}\\.){1,100}[a-z]{2,8}$"},
                    "minItems": 1,
                }
            },
            "required": ["domains"],
            "additionalProperties": False,
        }
        path, payload = chat_payload(
            args.model,
            "List two hundred plausible web domain names as JSON.",
            format_field("chat", "json_schema", domains),
            False,
            False,
            args.load_tokens,
        )
        started = time.monotonic()
        reply = post(args.base_url, path, payload, {}, args.timeout)
        seconds = time.monotonic() - started
        error = error_of(reply) if reply.status != 200 else {}
        finish = chat_extract(reply, False)[1] if reply.status == 200 else ""
        results.append(
            Result(
                f"cpu_load/domains_pattern_{args.cpu_hogs}_hogs",
                reply.status == 200 and finish in ("stop", "length"),
                f"HTTP {reply.status} finish={finish!r} code={error.get('code')!r}",
                seconds,
            )
        )
        name, outputs = AMBIGUOUS[1]
        path, payload = chat_payload(args.model, AMBIGUOUS_PROMPT, {"structured_outputs": outputs}, False, False, 3000)
        started = time.monotonic()
        reply = post(args.base_url, path, payload, {}, args.timeout)
        ok, error = ambiguous_outcome(name, reply, False)
        results.append(
            Result(
                f"cpu_load/{name}", ok, f"HTTP {reply.status} code={error.get('code')!r}", time.monotonic() - started
            )
        )
    finally:
        stop.set()
        for hog in hogs:
            hog.join(timeout=5)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--repeat", type=int, default=1, help="requests per valid-output case")
    parser.add_argument("--thinking", choices=["on", "off", "both"], default="both")
    parser.add_argument("--protocols", default="chat,responses,anthropic")
    parser.add_argument("--allow-port", type=int, action="append", default=[])
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument(
        "--robustness",
        choices=["on", "off", "only"],
        default="on",
        help="post-audit cases: thinking budget cutting characters, bounded whitespace, failing "
        "constrained requests next to free ones (uses the Anthropic endpoint)",
    )
    parser.add_argument("--no-hostile", action="store_true", help="skip the hostile-input checks (Chat)")
    parser.add_argument("--speed-tokens", type=int, default=1200, help="tokens of the concurrent speed probe")
    parser.add_argument("--speed-ratio", type=float, default=0.8, help="minimum speed next to hostile requests")
    parser.add_argument("--cpu-hogs", type=int, default=0, help="busy processes for the CPU-load checks (0 skips)")
    parser.add_argument("--load-tokens", type=int, default=1500, help="tokens of the CPU-load pattern request")
    args = parser.parse_args()
    args.thinking = {"on": [True], "off": [False], "both": [False, True]}[args.thinking]

    port = urlsplit(args.base_url).port
    if port in PRODUCTION_PORTS and port not in args.allow_port:
        print(f"refusing production port {port}; pass --allow-port {port} to override", file=sys.stderr)
        return 2
    if jsonschema is None:
        print("jsonschema is required: pip install -r tests/text/requirements.txt", file=sys.stderr)
        return 2

    results: list[Result] = []
    protocols = [p for p in args.protocols.split(",") if p] if args.robustness != "only" else []
    for protocol in protocols:
        results += check_unconstrained(args, protocol)
        for case in ("flat", "nested_enum", "array"):
            schema = SCHEMAS[case]
            results += check_valid(
                args, protocol, case, format_field(protocol, "json_schema", schema), json_validator(schema)
            )
        if protocol != "anthropic":
            results += check_valid(
                args, protocol, "json_object", format_field(protocol, "json_object", None), json_validator(None)
            )
        if protocol == "chat":
            results += check_valid(
                args,
                protocol,
                "choice",
                {"structured_outputs": {"choice": ["yes", "no"]}},
                lambda c: None if c in ("yes", "no") else f"not a choice: {c!r}",
            )
            results += check_valid(
                args,
                protocol,
                "regex",
                {"structured_outputs": {"regex": "[1-9][0-9]{2}"}},
                lambda c: None if re.fullmatch("[1-9][0-9]{2}", c) else f"regex mismatch: {c!r}",
            )
            results += check_valid(
                args,
                protocol,
                "grammar",
                {"structured_outputs": {"grammar": 'root ::= "red" | "green"'}},
                lambda c: None if c in ("red", "green") else f"grammar mismatch: {c!r}",
            )
    results += [r for r in check_errors(args) if r.name.split("/")[1] in protocols]
    if args.robustness != "off":
        results += check_robustness(args)
    if not args.no_hostile:
        results += check_deep_request(args, protocols)
    if "chat" in protocols:
        results += check_open_schema(args)
    if "chat" in protocols and not args.no_hostile:
        results += check_compilation_hostile(args)
    if "chat" in protocols and args.cpu_hogs > 0:
        results += check_cpu_load(args)
    if "chat" in protocols and not args.no_hostile:
        results += check_hostile_rejections(args)
        results += check_hostile_generation(args)

    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name:<{width}}  {r.seconds:6.2f}s  {r.detail[:140]}")
    failed = sum(not r.ok for r in results)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([r.__dict__ for r in results], handle, indent=2)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
