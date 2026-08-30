"""Audited request graphs for black-box ninfer-serve TTFT measurement."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from tools.bench.ttft.corpus import Corpus
from tools.bench.ttft.execution import (
    CaseContext,
    CaseExecutionError,
    FailedCondition,
    RequestHandle,
)
from tools.ninfer_serve.anthropic import anthropic_request
from tools.ninfer_serve.openai_chat import chat_request
from tools.ninfer_serve.openai_responses import responses_request


CaseFunction = Callable[[CaseContext, Corpus], None]
_CANCEL_TERMINATION_LIMIT_NS = 5_000_000_000


@dataclass(frozen=True)
class SymmetricRoleGroup:
    name: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    protocol: str
    profile: str
    category: str
    corpus_ids: tuple[str, ...]
    description: str
    run: CaseFunction
    symmetric_role_groups: tuple[SymmetricRoleGroup, ...] = ()


def _response_id(handle: RequestHandle) -> str:
    if not handle.response_id:
        raise CaseExecutionError(f"request {handle.role} completed without a response id")
    return handle.response_id


def _assistant(handle: RequestHandle) -> dict[str, str]:
    output = handle.output_text
    if not output:
        raise CaseExecutionError(f"request {handle.role} produced no assistant content")
    return {"role": "assistant", "content": output}


def _require_successes(context: CaseContext, handles: Iterable[RequestHandle]) -> None:
    for handle in handles:
        context.require_success(handle)


def _require_order(
    context: CaseContext,
    expression: str,
    values: Iterable[tuple[str, int | None]],
) -> None:
    observed = list(values)
    valid = all(isinstance(value, int) for _, value in observed)
    if valid:
        numeric = [int(value) for _, value in observed]
        valid = all(left < right for left, right in zip(numeric, numeric[1:]))
    detail = ", ".join(f"{name}={value}" for name, value in observed)
    context.require(valid, expression, detail)


def _chat_shape(context: CaseContext, corpus: Corpus, shape: str, role: str) -> RequestHandle:
    facts = corpus.shape(shape)
    return context.start(
        role,
        chat_request(
            context.model,
            corpus.shape_messages(shape),
            facts["max_output_tokens"],
        ),
    )


def _responses_shape(
    context: CaseContext,
    corpus: Corpus,
    shape: str,
    role: str,
    *,
    store: bool,
    previous_response_id: str | None = None,
) -> RequestHandle:
    facts = corpus.shape(shape)
    return context.start(
        role,
        responses_request(
            context.model,
            corpus.shape_messages(shape),
            facts["max_output_tokens"],
            store=store,
            previous_response_id=previous_response_id,
        ),
    )


def _cold_shape(shape: str) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        context.require_success(_chat_shape(context, corpus, shape, "request"))

    return run


def _anonymous_hot(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.shape_messages("long-8k-16")
    source = context.start(
        "source",
        chat_request(context.model, source_messages, 16),
    )
    context.require_success(source)
    continuation_messages = [*source_messages, _assistant(source)]
    continuation_messages.append(
        {"role": "user", "content": "Now answer with the saved code only."}
    )
    continuation = context.start(
        "continuation",
        chat_request(context.model, continuation_messages, 32),
    )
    context.require_success(continuation)


def _session_hot(context: CaseContext, corpus: Corpus) -> None:
    source = _responses_shape(
        context, corpus, "long-8k-16", "source", store=True
    )
    context.require_success(source)
    continuation = context.start(
        "continuation",
        responses_request(
            context.model,
            "Return the saved answer in one short line.",
            32,
            store=True,
            previous_response_id=_response_id(source),
        ),
    )
    context.require_success(continuation)


def _session_alternating(context: CaseContext, corpus: Corpus) -> None:
    messages_a = corpus.shape_messages("long-8k-16")
    messages_b = copy.deepcopy(messages_a)
    messages_b[-1]["content"] += "\nThis is independent session B."
    a1 = context.start(
        "a1", responses_request(context.model, messages_a, 16, store=True)
    )
    context.require_success(a1)
    b1 = context.start(
        "b1", responses_request(context.model, messages_b, 16, store=True)
    )
    context.require_success(b1)
    a2 = context.start(
        "a2",
        responses_request(
            context.model,
            "Continue session A briefly.",
            16,
            store=True,
            previous_response_id=_response_id(a1),
        ),
    )
    context.require_success(a2)
    b2 = context.start(
        "b2",
        responses_request(
            context.model,
            "Continue session B briefly.",
            16,
            store=True,
            previous_response_id=_response_id(b1),
        ),
    )
    context.require_success(b2)


def _session_alternating_64k_host_swap(context: CaseContext, corpus: Corpus) -> None:
    a1 = _responses_shape(
        context, corpus, "long-64k-32", "a1", store=True
    )
    context.require_success(a1)
    b1 = _responses_shape(
        context, corpus, "long-64k-independent-32", "b1", store=True
    )
    context.require_success(b1)
    a2 = context.start(
        "a2",
        responses_request(
            context.model,
            "Continue session A briefly.",
            32,
            store=True,
            previous_response_id=_response_id(a1),
        ),
    )
    context.require_success(a2)
    b2 = context.start(
        "b2",
        responses_request(
            context.model,
            "Continue session B briefly.",
            32,
            store=True,
            previous_response_id=_response_id(b1),
        ),
    )
    context.require_success(b2)


def _unmarked_common(context: CaseContext, corpus: Corpus) -> None:
    first = _chat_shape(context, corpus, "unmarked-common-a", "first")
    context.require_success(first)
    second = _chat_shape(context, corpus, "unmarked-common-b", "second")
    context.require_success(second)


def _pressure_graph(context: CaseContext, corpus: Corpus) -> None:
    source = _responses_shape(
        context, corpus, "long-8k-16", "source", store=True
    )
    context.require_success(source)
    interferer_messages = corpus.shape_messages("interferer-256")
    interferers = context.barrier(
        (
            (
                "interferer-b",
                responses_request(
                    context.model, interferer_messages, 256, store=False
                ),
            ),
            (
                "interferer-c",
                responses_request(
                    context.model, interferer_messages, 256, store=False
                ),
            ),
        )
    )
    for handle in interferers:
        handle.wait_first_output(context.timeout)
    _require_successes(context, interferers)
    latest_first = max(int(handle.first_output_ns) for handle in interferers)
    earliest_complete = min(int(handle.completed_ns) for handle in interferers)
    context.require(
        latest_first < earliest_complete,
        "max(B.first,C.first) < min(B.completed,C.completed)",
        f"latest_first={latest_first}, earliest_complete={earliest_complete}",
    )
    resume = context.start(
        "resume",
        responses_request(
            context.model,
            "Return the retained answer in one line.",
            32,
            store=True,
            previous_response_id=_response_id(source),
        ),
    )
    context.require_success(resume)


def _cache_off(context: CaseContext, corpus: Corpus) -> None:
    _session_hot(context, corpus)


def _anthropic_system(
    context: CaseContext,
    system: str,
    suffix: str,
    role: str,
) -> RequestHandle:
    return context.start(
        role,
        anthropic_request(
            context.model,
            [{"role": "user", "content": suffix}],
            32,
            system=system,
        ),
    )


def _shared_sequential(context: CaseContext, corpus: Corpus) -> None:
    system = corpus.shared_system("system-a")
    first = _anthropic_system(context, system, "Suffix one: answer briefly.", "first")
    context.require_success(first)
    second = _anthropic_system(context, system, "Suffix two: answer briefly.", "second")
    context.require_success(second)


def _openai_explicit_shared(context: CaseContext, corpus: Corpus) -> None:
    system = [
        {
            "type": "text",
            "text": corpus.shared_system("system-a"),
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    first = context.start(
        "first",
        chat_request(
            context.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "Suffix one: answer briefly."},
            ],
            32,
        ),
    )
    context.require_success(first)
    second = context.start(
        "second",
        chat_request(
            context.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "Suffix two: answer briefly."},
            ],
            32,
        ),
    )
    context.require_success(second)


def _openai_implicit_shared(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.shape_messages("unmarked-common-a")
    facts = corpus.shape("unmarked-common-a")
    first = context.start(
        "first", chat_request(context.model, messages, facts["max_output_tokens"])
    )
    context.require_success(first)
    filler = context.start(
        "private-filler",
        chat_request(
            context.model,
            [{"role": "user", "content": "Independent private filler."}],
            32,
        ),
    )
    context.require_success(filler)
    reuse = context.start(
        "reuse", chat_request(context.model, messages, facts["max_output_tokens"])
    )
    context.require_success(reuse)


def _observed_shared_promotion(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.shape_messages("unmarked-common-a")
    facts = corpus.shape("unmarked-common-a")
    first = context.start(
        "first-observation",
        anthropic_request(context.model, messages, facts["max_output_tokens"]),
    )
    context.require_success(first)
    promotion = context.start(
        "promotion",
        anthropic_request(context.model, messages, facts["max_output_tokens"]),
    )
    context.require_success(promotion)
    filler = context.start(
        "private-filler",
        anthropic_request(
            context.model,
            [{"role": "user", "content": "Independent private filler."}],
            32,
        ),
    )
    context.require_success(filler)
    reuse = context.start(
        "shared-reuse",
        anthropic_request(context.model, messages, facts["max_output_tokens"]),
    )
    context.require_success(reuse)


def _shared_fanout(context: CaseContext, corpus: Corpus) -> None:
    system = corpus.shared_system("system-a")
    seed = _anthropic_system(context, system, "Establish this stable prefix.", "seed")
    context.require_success(seed)
    branches = context.barrier(
        (
            (
                "branch-b",
                anthropic_request(
                    context.model,
                    [{"role": "user", "content": "Independent branch B."}],
                    32,
                    system=system,
                ),
            ),
            (
                "branch-c",
                anthropic_request(
                    context.model,
                    [{"role": "user", "content": "Independent branch C."}],
                    32,
                    system=system,
                ),
            ),
        )
    )
    _require_successes(context, branches)


def _shared_slot_retention(context: CaseContext, corpus: Corpus) -> None:
    system_a = corpus.shared_system("system-a")
    system_b = corpus.shared_system("system-b")
    first_a = _anthropic_system(context, system_a, "Prefix A first use.", "a-first")
    context.require_success(first_a)
    first_b = _anthropic_system(context, system_b, "Prefix B competes with A.", "b")
    context.require_success(first_b)
    filler = context.start(
        "private-filler",
        anthropic_request(
            context.model,
            [{"role": "user", "content": "Unmarked private filler request."}],
            32,
        ),
    )
    context.require_success(filler)
    second_a = _anthropic_system(context, system_a, "Prefix A after competition.", "a-final")
    context.require_success(second_a)


def _shared_value_replacement(context: CaseContext, corpus: Corpus) -> None:
    system = corpus.shared_system("system-a")
    first_a = _anthropic_system(context, system, "Establish prefix A.", "a")
    context.require_success(first_a)
    tools = corpus.client_tools()
    first_b = context.start(
        "b-first",
        anthropic_request(
            context.model,
            [
                {
                    "role": "user",
                    "content": "Do not call a tool. Summarize what these tools can inspect.",
                }
            ],
            32,
            tools=tools,
        ),
    )
    context.require_success(first_b)
    filler = context.start(
        "private-filler",
        anthropic_request(
            context.model,
            [{"role": "user", "content": "Independent private filler."}],
            32,
        ),
    )
    context.require_success(filler)
    second_b = context.start(
        "b-reuse",
        anthropic_request(
            context.model,
            [
                {
                    "role": "user",
                    "content": "Do not call a tool. Name one safe repository operation.",
                }
            ],
            32,
            tools=tools,
        ),
    )
    context.require_success(second_b)


def _shared_tools(context: CaseContext, corpus: Corpus, *, changed: bool) -> None:
    tools = corpus.client_tools()
    first = context.start(
        "first",
        anthropic_request(
            context.model,
            [{"role": "user", "content": "Do not call a tool. Summarize what these tools can inspect."}],
            32,
            tools=tools,
        ),
    )
    context.require_success(first)
    second_tools = corpus.client_tools(changed_first=changed)
    second = context.start(
        "second",
        anthropic_request(
            context.model,
            [{"role": "user", "content": "Do not call a tool. Name one safe repository operation."}],
            32,
            tools=second_tools,
        ),
    )
    context.require_success(second)


def _shared_tools_sequential(context: CaseContext, corpus: Corpus) -> None:
    _shared_tools(context, corpus, changed=False)


def _shared_tools_changed(context: CaseContext, corpus: Corpus) -> None:
    _shared_tools(context, corpus, changed=True)


def _short_during_prefill(context: CaseContext, corpus: Corpus) -> None:
    long_request = _chat_shape(context, corpus, "long-8k-32", "long")
    long_request.wait_accepted(context.timeout)
    short = _chat_shape(context, corpus, "short-32", "short")
    _require_successes(context, (long_request, short))
    _require_order(
        context,
        "long.accepted < short.sent < long.first",
        (
            ("long.accepted", long_request.accepted_ns),
            ("short.sent", short.sent_ns),
            ("long.first", long_request.first_output_ns),
        ),
    )


def _short_during_decode(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    short = _chat_shape(context, corpus, "short-32", "short")
    _require_successes(context, (holder, short))
    _require_order(
        context,
        "holder.first < short.sent < holder.completed",
        (
            ("holder.first", holder.first_output_ns),
            ("short.sent", short.sent_ns),
            ("holder.completed", holder.completed_ns),
        ),
    )


def _protected_backfill(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    head = _chat_shape(context, corpus, "long-8k-32", "head")
    head.wait_accepted(context.timeout)
    borrower = _chat_shape(context, corpus, "short-32", "borrower")
    _require_successes(context, (holder, head, borrower))
    _require_order(
        context,
        "head.accepted < borrower.sent < borrower.first < holder.completed < head.first",
        (
            ("head.accepted", head.accepted_ns),
            ("borrower.sent", borrower.sent_ns),
            ("borrower.first", borrower.first_output_ns),
            ("holder.completed", holder.completed_ns),
            ("head.first", head.first_output_ns),
        ),
    )


def _protected_no_backfill(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    head = _chat_shape(context, corpus, "long-8k-32", "head")
    head.wait_accepted(context.timeout)
    borrower = _chat_shape(context, corpus, "medium-3000", "unsafe-borrower")
    _require_successes(context, (holder, head, borrower))
    _require_order(
        context,
        "head.accepted < borrower.sent < holder.completed < head.first < head.completed < borrower.first",
        (
            ("head.accepted", head.accepted_ns),
            ("borrower.sent", borrower.sent_ns),
            ("holder.completed", holder.completed_ns),
            ("head.first", head.first_output_ns),
            ("head.completed", head.completed_ns),
            ("borrower.first", borrower.first_output_ns),
        ),
    )


def _active_lanes_full(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.shape_messages("holder-4096")
    holders = context.barrier(
        (
            (f"holder-{index}", chat_request(context.model, messages, 4096))
            for index in range(8)
        )
    )
    for holder in holders:
        holder.wait_first_output(context.timeout)
    probe = _chat_shape(context, corpus, "short-32", "probe")
    _require_successes(context, [*holders, probe])
    earliest_holder_completion = min(int(item.completed_ns) for item in holders)
    context.require(
        isinstance(probe.sent_ns, int)
        and isinstance(probe.first_output_ns, int)
        and probe.sent_ns < earliest_holder_completion <= probe.first_output_ns,
        "probe.sent < min(holder.completed) <= probe.first",
        f"probe.sent={probe.sent_ns}, earliest_holder_completion={earliest_holder_completion}, probe.first={probe.first_output_ns}",
    )


def _session_publication_order(context: CaseContext, corpus: Corpus) -> None:
    parent = context.start(
        "parent",
        responses_request(context.model, "Create a session root.", 16, store=True),
    )
    context.require_success(parent)
    parent_id = _response_id(parent)
    older = context.start(
        "older",
        responses_request(
            context.model,
            corpus.shape_messages("holder-4096"),
            4096,
            store=True,
            previous_response_id=parent_id,
        ),
    )
    older.wait_first_output(context.timeout)
    newer = context.start(
        "newer",
        responses_request(
            context.model,
            "Finish this newer branch immediately.",
            16,
            store=True,
            previous_response_id=parent_id,
        ),
    )
    context.require_success(newer)
    context.require_success(older)
    _require_order(
        context,
        "older.first < newer.sent < newer.completed < older.completed",
        (
            ("older.first", older.first_output_ns),
            ("newer.sent", newer.sent_ns),
            ("newer.completed", newer.completed_ns),
            ("older.completed", older.completed_ns),
        ),
    )
    fillers: list[RequestHandle] = []
    for index in range(3):
        filler = context.start(
            f"pressure-{index}",
            responses_request(
                context.model,
                f"Independent stored pressure root {index}.",
                16,
                store=True,
            ),
        )
        context.require_success(filler)
        fillers.append(filler)
    final = context.start(
        "continue-newer",
        responses_request(
            context.model,
            "Continue the newest child.",
            16,
            store=True,
            previous_response_id=_response_id(newer),
        ),
    )
    context.require_success(final)


def _require_transport_cancellation(
    context: CaseContext,
    handle: RequestHandle,
    cancel_ns: int,
) -> None:
    result = handle.result
    completed_ns = handle.completed_ns
    terminal_events = (
        [event for event in result.events if event.kind in {"terminal", "error"}]
        if result is not None
        else []
    )
    context.require(
        handle.outcome() == "cancelled"
        and result is not None
        and result.http.cancel_requested
        and result.http.cancelled,
        f"{handle.role}.transport terminated by cancellation",
        (
            f"outcome={handle.outcome()}, "
            f"cancel_requested={result.http.cancel_requested if result else None}, "
            f"transport_cancelled={result.http.cancelled if result else None}"
        ),
    )
    context.require(
        not terminal_events,
        f"{handle.role} has no protocol terminal after cancellation",
        f"terminal_events={[event.event_type for event in terminal_events]}",
    )
    context.require(
        isinstance(completed_ns, int)
        and cancel_ns <= completed_ns <= cancel_ns + _CANCEL_TERMINATION_LIMIT_NS,
        f"0 <= {handle.role}.completed-{handle.role}.cancel <= 5s",
        f"cancel={cancel_ns}, completed={completed_ns}",
    )


def _cancel_before_first(context: CaseContext, corpus: Corpus) -> None:
    request = _chat_shape(context, corpus, "long-8k-32", "cancelled")
    request.wait_accepted(context.timeout)
    cancel_ns = request.cancel()
    request.wait_done(context.timeout)
    probe = _chat_shape(context, corpus, "short-32", "probe")
    context.require_success(probe)
    context.require(
        request.first_output_ns is None,
        "cancelled.first is absent",
        f"cancel={cancel_ns}, first={request.first_output_ns}",
    )
    _require_transport_cancellation(context, request, cancel_ns)


def _cancel_after_first(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "cancelled-holder")
    holder.wait_first_output(context.timeout)
    cancel_ns = holder.cancel()
    holder.wait_done(context.timeout)
    probe = _chat_shape(context, corpus, "short-32", "probe")
    context.require_success(probe)
    _require_order(
        context,
        "holder.first < holder.cancel < holder.completed",
        (
            ("holder.first", holder.first_output_ns),
            ("holder.cancel", cancel_ns),
            ("holder.completed", holder.completed_ns),
        ),
    )
    _require_transport_cancellation(context, holder, cancel_ns)


def _expect_rejection(
    context: CaseContext,
    handle: RequestHandle,
    status: int,
    code: str,
) -> None:
    handle.wait_done(context.timeout)
    record = handle.as_record()
    context.require(
        record["http_status"] == status and record["error_code"] == code,
        f"{handle.role}.status/code == {status}/{code}",
        f"status={record['http_status']}, code={record['error_code']!r}, outcome={record['outcome']}",
    )
    context.require(
        record["ttft_ns"] is None,
        f"{handle.role}.ttft is absent",
        f"ttft_ns={record['ttft_ns']}",
    )


def _pending_overflow(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.shape_messages("holder-4096")
    holders = context.barrier(
        (
            (f"holder-{index}", chat_request(context.model, messages, 4096))
            for index in range(8)
        )
    )
    for holder in holders:
        holder.wait_first_output(context.timeout)
    pending = _chat_shape(context, corpus, "short-32", "pending")
    pending.wait_accepted(context.timeout)
    overflow = _chat_shape(context, corpus, "short-32", "overflow")
    _expect_rejection(context, overflow, 429, "server_overloaded")
    for holder in holders:
        holder.cancel()
    for holder in holders:
        holder.wait_done(context.timeout)
    context.require_success(pending)


def _pending_timeout(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    messages = corpus.shape_messages("short-32")
    timeout_probe = context.start(
        "timeout-probe",
        chat_request(context.model, messages, 32, stream=False),
    )
    _expect_rejection(context, timeout_probe, 503, "request_queue_timeout")
    holder.cancel()
    holder.wait_done(context.timeout)


def _context_exact(context: CaseContext, corpus: Corpus) -> None:
    context.require_success(_chat_shape(context, corpus, "context-exact", "request"))


def _context_over(context: CaseContext, corpus: Corpus) -> None:
    request = context.start(
        "request",
        chat_request(context.model, corpus.shape_messages("context-over"), 16),
    )
    _expect_rejection(context, request, 400, "context_length_exceeded")


def _mixed_four_requests(
    context: CaseContext,
    corpus: Corpus,
) -> list[tuple[str, Any]]:
    source_messages = corpus.shape_messages("long-8k-16")
    seed = context.start("seed", chat_request(context.model, source_messages, 16))
    context.require_success(seed)
    continuation_messages = [*source_messages, _assistant(seed)]
    continuation_messages.append({"role": "user", "content": "Continue briefly."})
    return [
        (
            "continuation",
            chat_request(context.model, continuation_messages, 32),
        ),
        (
            "cold-long",
            chat_request(
                context.model,
                corpus.shape_messages("long-8k-independent-32"),
                32,
            ),
        ),
        (
            "short",
            chat_request(context.model, corpus.shape_messages("short-32"), 32),
        ),
        (
            "image",
            chat_request(context.model, corpus.media_messages("image-chart"), 32),
        ),
    ]


def _mixed_four_ordered(context: CaseContext, corpus: Corpus) -> None:
    handles = [
        context.prepare(role, request)
        for role, request in _mixed_four_requests(context, corpus)
    ]
    for handle in handles:
        handle.start()
        handle.wait_accepted(context.timeout)
    _require_successes(context, handles)
    accepted = [int(handle.accepted_ns) for handle in handles]
    context.require(
        all(left < right for left, right in zip(accepted, accepted[1:])),
        "continuation.accepted < cold-long.accepted < short.accepted < image.accepted",
        f"accepted={accepted}",
    )


def _mixed_four_concurrent(context: CaseContext, corpus: Corpus) -> None:
    handles = context.barrier(_mixed_four_requests(context, corpus))
    _require_successes(context, handles)


def _media_cold(name: str) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        request = context.start(
            "request",
            chat_request(context.model, corpus.media_messages(name), 32),
        )
        context.require_success(request)

    return run


def _replace_first_image(messages: list[dict[str, Any]], replacement_url: str) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["image_url"] = {"url": replacement_url}
                return
    raise CaseExecutionError("media history has no image to replace")


def _media_prefix_continuation(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.media_messages("image-chart")
    source = context.start("source", chat_request(context.model, source_messages, 32))
    context.require_success(source)
    history = [*source_messages, _assistant(source)]
    history.append({"role": "user", "content": "Recall the same image in one phrase."})
    continuation = context.start(
        "continuation", chat_request(context.model, history, 32)
    )
    context.require_success(continuation)


def _media_prefix_append(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.media_messages("image-chart")
    source = context.start("source", chat_request(context.model, source_messages, 32))
    context.require_success(source)
    history = [*source_messages, _assistant(source)]
    history.extend(corpus.load_image_messages(1, prompt="Compare this new image with the prior one."))
    continuation = context.start(
        "continuation", chat_request(context.model, history, 32)
    )
    context.require_success(continuation)


def _media_prefix_changed(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.media_messages("image-chart")
    source = context.start("source", chat_request(context.model, source_messages, 32))
    context.require_success(source)
    changed = copy.deepcopy(source_messages)
    replacement = corpus.load_image_messages(2)[0]["content"][0]["image_url"]["url"]
    _replace_first_image(changed, replacement)
    changed.append(_assistant(source))
    changed.append({"role": "user", "content": "Answer after the earlier image changed."})
    continuation = context.start(
        "changed-continuation", chat_request(context.model, changed, 32)
    )
    context.require_success(continuation)


def _media_preprocess_warm(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.load_image_messages(0)
    first = context.start("first", chat_request(context.model, messages, 32))
    context.require_success(first)
    second = context.start("second", chat_request(context.model, messages, 32))
    context.require_success(second)


def _media_cache_thrash(context: CaseContext, corpus: Corpus) -> None:
    handles = []
    for role, index in (("a-first", 0), ("b", 1), ("c", 2), ("a-final", 0)):
        handle = context.start(
            role,
            chat_request(context.model, corpus.load_image_messages(index), 32),
        )
        context.require_success(handle)
        handles.append(handle)


def _many_image(context: CaseContext, corpus: Corpus) -> None:
    request = context.start(
        "request",
        chat_request(context.model, corpus.media_messages("many-image-28-a"), 32),
    )
    context.require_success(request)


def _text_during_media_prepare(context: CaseContext, corpus: Corpus) -> None:
    short = context.prepare(
        "short",
        chat_request(context.model, corpus.shape_messages("short-32"), 32),
    )
    media = context.prepare(
        "media",
        chat_request(
            context.model, corpus.media_messages("many-image-28-a"), 32
        ),
    )
    media.start()
    media.wait_body_sent(context.timeout)
    short.start()
    _require_successes(context, (media, short))
    _require_order(
        context,
        "media.body_sent < short.sent < media.accepted",
        (
            ("media.body_sent", media.body_sent_ns),
            ("short.sent", short.sent_ns),
            ("media.accepted", media.accepted_ns),
        ),
    )
    _require_order(
        context,
        "short.first < media.first",
        (("short.first", short.first_output_ns), ("media.first", media.first_output_ns)),
    )


def _media_during_text_decode(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    media = context.start(
        "media",
        chat_request(context.model, corpus.media_messages("many-image-28-a"), 32),
    )
    _require_successes(context, (holder, media))
    _require_order(
        context,
        "holder.first < media.sent < holder.completed",
        (
            ("holder.first", holder.first_output_ns),
            ("media.sent", media.sent_ns),
            ("holder.completed", holder.completed_ns),
        ),
    )


def _two_heavy_media(context: CaseContext, corpus: Corpus) -> None:
    handles = context.barrier(
        (
            (
                "media-a",
                chat_request(
                    context.model, corpus.media_messages("many-image-28-a"), 32
                ),
            ),
            (
                "media-b",
                chat_request(
                    context.model, corpus.media_messages("many-image-28-b"), 32
                ),
            ),
        )
    )
    _require_successes(context, handles)


def _vision_disabled(context: CaseContext, corpus: Corpus) -> None:
    request = context.start(
        "request",
        chat_request(context.model, corpus.media_messages("image-chart"), 32),
    )
    _expect_rejection(context, request, 400, "vision_disabled")


def _vision_envelope_over(context: CaseContext, corpus: Corpus) -> None:
    request = context.start(
        "request",
        chat_request(context.model, corpus.media_messages("many-image-33"), 32),
    )
    _expect_rejection(context, request, 400, "media_budget_exceeded")


def _definition(
    name: str,
    protocol: str,
    profile: str,
    category: str,
    corpus_ids: tuple[str, ...],
    description: str,
    run: CaseFunction,
    *,
    symmetric_role_groups: tuple[SymmetricRoleGroup, ...] = (),
) -> CaseDefinition:
    return CaseDefinition(
        name,
        protocol,
        profile,
        category,
        corpus_ids,
        description,
        run,
        symmetric_role_groups,
    )


_DEFINITIONS = (
    _definition("cold-short", "openai_chat", "text-cold-8k", "workload", ("short-32",), "Short cold TTFT baseline.", _cold_shape("short-32")),
    _definition("cold-long-8k", "openai_chat", "text-cold-8k", "workload", ("long-8k-32",), "8K cold prefill baseline.", _cold_shape("long-8k-32")),
    _definition("cold-long-64k", "openai_chat", "text-cold-64k", "workload", ("long-64k-32",), "64K legal long-context input.", _cold_shape("long-64k-32")),
    _definition("cold-long-256k", "openai_chat", "text-cold-256k", "workload", ("long-256k-32",), "Hardware-resident extreme context under the standard FP8 KV profile.", _cold_shape("long-256k-32")),
    _definition("mixed-four-ordered", "openai_chat", "mixed-four", "workload", ("long-8k-16", "long-8k-independent-32", "short-32", "image-chart"), "Four legal heterogeneous arrivals submitted in observed Serve-acceptance order.", _mixed_four_ordered),
    _definition("mixed-four-concurrent", "openai_chat", "mixed-four", "workload", ("long-8k-16", "long-8k-independent-32", "short-32", "image-chart"), "The same heterogeneous requests released concurrently without an artificial order.", _mixed_four_concurrent),
    _definition("anonymous-hot-continuation", "openai_chat", "cache-hot", "private", ("long-8k-16",), "Anonymous full-history continuation.", _anonymous_hot),
    _definition("session-hot-continuation", "openai_responses", "cache-hot", "session", ("long-8k-16",), "Named Responses session continuation.", _session_hot),
    _definition("session-alternating", "openai_responses", "cache-pressure-device", "session", ("long-8k-16",), "Alternating named sessions.", _session_alternating),
    _definition("session-alternating-64k-host-swap", "openai_responses", "cache-swap-64k-host", "resource", ("long-64k-32", "long-64k-independent-32"), "Two near-capacity sessions alternate through Host KV.", _session_alternating_64k_host_swap),
    _definition("unmarked-common-prefix-miss", "openai_chat", "cache-hot", "private", ("unmarked-common-a", "unmarked-common-b"), "Raw token commonality without a checkpoint.", _unmarked_common),
    _definition("resume-after-interference-device", "openai_responses", "cache-pressure-device", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Device-resident source.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-state-host", "openai_responses", "cache-pressure-state-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Host State restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-kv-host", "openai_responses", "cache-pressure-kv-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Host KV restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-both-host", "openai_responses", "cache-pressure-both-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with State and KV restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-evicted", "openai_responses", "cache-pressure-evict", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with no legal retained replica.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-catalog", "openai_responses", "cache-pressure-catalog", "resource", ("long-8k-16", "interferer-256"), "Descriptor pressure with spare physical capacity.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("continuation-cache-off", "openai_responses", "cache-off", "control", ("long-8k-16",), "Full-history Responses control with Engine reuse disabled.", _cache_off),
    _definition("shared-sequential", "anthropic_messages", "shared-prefix", "shared", ("system-a",), "Marked Anthropic shared prefix sequential reuse.", _shared_sequential),
    _definition("shared-openai-explicit", "openai_chat", "shared-prefix", "shared", ("system-a",), "OpenAI explicit system boundary shared by different suffixes.", _openai_explicit_shared),
    _definition("shared-openai-implicit", "openai_chat", "shared-value", "shared", ("unmarked-common-a",), "OpenAI default implicit full-prompt owner survives private displacement.", _openai_implicit_shared),
    _definition("shared-observed-promotion", "anthropic_messages", "shared-value", "shared", ("unmarked-common-a",), "Two independent unmarked observations promote a private base before private displacement.", _observed_shared_promotion),
    _definition("shared-fanout", "anthropic_messages", "shared-prefix", "shared", ("system-a",), "Concurrent branches from a non-aligned shared prefix.", _shared_fanout, symmetric_role_groups=(SymmetricRoleGroup("branches", ("branch-b", "branch-c")),)),
    _definition("shared-slot-retention", "anthropic_messages", "shared-value", "shared", ("system-a", "system-b"), "Equal-value S=1 challenger must justify replacing the established owner.", _shared_slot_retention),
    _definition("shared-value-replacement", "anthropic_messages", "shared-value", "shared", ("system-a", "client-tools-32"), "A higher-value tool frontier competes for one shared slot.", _shared_value_replacement),
    _definition("shared-tools-sequential", "anthropic_messages", "shared-prefix", "shared", ("client-tools-32",), "Stable 32-tool prefix reuse.", _shared_tools_sequential),
    _definition("shared-tools-changed", "anthropic_messages", "shared-prefix", "shared", ("client-tools-32",), "Early tool identity change invalidates the marked prefix.", _shared_tools_changed),
    _definition("short-during-prefill-128", "openai_chat", "scheduler-prefill-128", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 128-token chunks.", _short_during_prefill),
    _definition("short-during-prefill-1024", "openai_chat", "scheduler-overlap", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 1024-token chunks.", _short_during_prefill),
    _definition("short-during-prefill-4096", "openai_chat", "scheduler-prefill-4096", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 4096-token chunks.", _short_during_prefill),
    _definition("short-during-decode", "openai_chat", "scheduler-overlap", "scheduling", ("holder-4096", "short-32"), "New prefill during active decode.", _short_during_decode),
    _definition("protected-head-backfill", "openai_chat", "scheduler-backfill", "scheduling", ("holder-4096", "long-8k-32", "short-32"), "Persistent-safe borrower may pass a protected head.", _protected_backfill),
    _definition("protected-head-no-backfill", "openai_chat", "scheduler-backfill", "scheduling", ("holder-4096", "long-8k-32", "medium-3000"), "Unsafe borrower stays behind a protected head.", _protected_no_backfill),
    _definition("active-lanes-full-8", "openai_chat", "lane-limit-8", "boundary", ("holder-4096", "short-32"), "Startup-fixed eight-lane product boundary.", _active_lanes_full, symmetric_role_groups=(SymmetricRoleGroup("holders", tuple(f"holder-{index}" for index in range(8))),)),
    _definition("session-publication-order", "openai_responses", "session-order", "boundary", ("holder-4096",), "Older late completion cannot overwrite newer session binding.", _session_publication_order),
    _definition("cancel-before-first", "openai_chat", "scheduler-overlap", "boundary", ("long-8k-32", "short-32"), "Cancellation after admission but before first output.", _cancel_before_first),
    _definition("cancel-after-first", "openai_chat", "scheduler-overlap", "boundary", ("holder-4096", "short-32"), "Active cancellation after first output.", _cancel_after_first),
    _definition("pending-overflow", "openai_chat", "lane-limit-8", "rejection", ("holder-4096", "short-32"), "Bounded FIFO overflow returns 429 without TTFT.", _pending_overflow),
    _definition("pending-timeout", "openai_chat", "pending-timeout", "rejection", ("holder-4096", "short-32"), "Non-streaming pending request returns 503.", _pending_timeout),
    _definition("context-exact", "openai_chat", "context-boundary", "boundary", ("context-exact",), "p+(o-1) exactly equals max_context.", _context_exact),
    _definition("context-over", "openai_chat", "context-boundary", "rejection", ("context-over",), "Prompt beyond max_context returns 400 without TTFT.", _context_over),
    _definition("media-cold-image", "openai_chat", "vision-cache", "media", ("image-chart",), "Single-image external TTFT.", _media_cold("image-chart")),
    _definition("media-cold-image-video", "openai_chat", "vision-cache", "media", ("image-video",), "Interleaved image and video input.", _media_cold("image-video")),
    _definition("media-prefix-continuation", "openai_chat", "vision-cache", "media", ("image-chart",), "Exact multimodal prefix continuation.", _media_prefix_continuation),
    _definition("media-prefix-append", "openai_chat", "vision-cache", "media", ("image-chart", "load-image-01"), "Reused media prefix with new suffix image.", _media_prefix_append),
    _definition("media-prefix-changed", "openai_chat", "vision-cache", "media", ("image-chart", "load-image-02"), "Earlier media identity change invalidates reuse.", _media_prefix_changed),
    _definition("media-preprocess-warm", "openai_chat", "media-cache-tight", "media", ("load-image-00",), "Media preprocess cache warm control with context reuse off.", _media_preprocess_warm),
    _definition("media-cache-thrash", "openai_chat", "media-cache-tight", "media", ("load-image-00", "load-image-01", "load-image-02"), "A-B-C-A media LRU pressure.", _media_cache_thrash),
    _definition("many-image-28", "openai_chat", "vision-cache", "media", ("many-image-28-a",), "Legal high Vision and many-item request.", _many_image),
    _definition("many-image-28-thread-1", "openai_chat", "vision-thread-1", "media", ("many-image-28-a",), "Single preprocessing worker control.", _many_image),
    _definition("text-during-media-prepare", "openai_chat", "vision-concurrent", "media", ("many-image-28-a", "short-32"), "Text arrival while a complete media body is preparing.", _text_during_media_prepare),
    _definition("media-during-text-decode", "openai_chat", "vision-concurrent", "media", ("holder-4096", "many-image-28-a"), "Heavy media arrival during text decode.", _media_during_text_decode),
    _definition("two-heavy-media-arrivals", "openai_chat", "vision-concurrent", "media", ("many-image-28-a", "many-image-28-b"), "Two byte-distinct legal high-media arrivals.", _two_heavy_media, symmetric_role_groups=(SymmetricRoleGroup("media", ("media-a", "media-b")),)),
    _definition("vision-disabled", "openai_chat", "text-cold-8k", "rejection", ("image-chart",), "Media on a text-only Serve is rejected.", _vision_disabled),
    _definition("vision-envelope-over", "openai_chat", "vision-boundary", "rejection", ("many-image-33",), "Aggregate Vision envelope rejection.", _vision_envelope_over),
)


CASES: dict[str, CaseDefinition] = {case.name: case for case in _DEFINITIONS}
if len(CASES) != len(_DEFINITIONS):
    raise RuntimeError("duplicate Serve TTFT case name")


def get_case(name: str) -> CaseDefinition:
    try:
        return CASES[name]
    except KeyError as error:
        raise CaseExecutionError(f"unknown TTFT case: {name}") from error


def run_case(
    definition: CaseDefinition,
    context: CaseContext,
    corpus: Corpus | None,
    profile_label: str,
) -> dict[str, Any]:
    if profile_label != definition.profile:
        return {
            "artifact_type": "ninfer_serve_ttft_run",
            "schema_version": 1,
            "case": definition.name,
            "protocol": definition.protocol,
            "category": definition.category,
            "description": definition.description,
            "profile_label": profile_label,
            "expected_profile_label": definition.profile,
            "corpus_ids": list(definition.corpus_ids),
            "symmetric_role_groups": [
                {"name": group.name, "roles": list(group.roles)}
                for group in definition.symmetric_role_groups
            ],
            "status": "invalid_profile_provenance",
            "constructed": False,
            "failed_conditions": [
                {
                    "expression": "profile_label == expected_profile_label",
                    "detail": f"{profile_label!r} != {definition.profile!r}",
                }
            ],
            "requests": [],
        }

    if corpus is None:
        raise CaseExecutionError("matching profile execution requires a corpus")

    try:
        definition.run(context, corpus)
        context.wait_all()
    except CaseExecutionError as error:
        context.progress("case.execution_failed", error=str(error))
        context.failures.append(FailedCondition("case execution completed", str(error)))
    except Exception as error:
        context.progress(
            "case.execution_failed",
            error=f"{type(error).__name__}: {error}",
        )
        context.failures.append(
            FailedCondition("case execution completed", f"{type(error).__name__}: {error}")
        )
    finally:
        context.cancel_live()

    if not context.failures and any(handle.outcome() == "running" for handle in context.handles):
        context.failures.append(FailedCondition("all requests terminated", "live request remains"))

    records = context.records()
    first_order = [
        record["role"]
        for record in sorted(
            (record for record in records if record["first_output_ns"] is not None),
            key=lambda record: record["first_output_ns"],
        )
    ]
    completion_order = [
        record["role"]
        for record in sorted(
            (record for record in records if record["completed_ns"] is not None),
            key=lambda record: record["completed_ns"],
        )
    ]
    constructed = not context.failures
    context.progress(
        "case.graph_finished",
        constructed=constructed,
        requests=len(records),
        failures=len(context.failures),
    )
    return {
        "artifact_type": "ninfer_serve_ttft_run",
        "schema_version": 1,
        "case": definition.name,
        "protocol": definition.protocol,
        "category": definition.category,
        "description": definition.description,
        "profile_label": profile_label,
        "expected_profile_label": definition.profile,
        "corpus_ids": list(definition.corpus_ids),
        "status": "constructed" if constructed else "not_constructed",
        "constructed": constructed,
        "failed_conditions": [failure.as_json() for failure in context.failures],
        "first_output_order": first_order,
        "completion_order": completion_order,
        "notes": context.notes,
        "symmetric_role_groups": [
            {"name": group.name, "roles": list(group.roles)}
            for group in definition.symmetric_role_groups
        ],
        "requests": records,
    }
