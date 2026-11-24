"""Audited request graphs for black-box ninfer-serve TTFT measurement."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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


def _mark_generated_input(context: CaseContext) -> None:
    context.notes["input_dependency"] = "generated"
    context.notes["throughput_comparable"] = False
    reason = "Later inputs include generated assistant output or stored response history."
    existing = context.notes.get("throughput_limitation", "")
    if reason not in existing:
        context.notes["throughput_limitation"] = " ".join(filter(None, (existing, reason)))


def _response_id(context: CaseContext, handle: RequestHandle) -> str:
    if not handle.response_id:
        raise CaseExecutionError(f"request {handle.role} completed without a response id")
    _mark_generated_input(context)
    return handle.response_id


def _assistant(context: CaseContext, handle: RequestHandle) -> dict[str, str]:
    output = handle.output_text
    if not output:
        raise CaseExecutionError(f"request {handle.role} produced no assistant content")
    _mark_generated_input(context)
    return {"role": "assistant", "content": output}


def _require_successes(context: CaseContext, handles: Iterable[RequestHandle]) -> None:
    for handle in handles:
        context.require_success(handle, prerequisite=False)


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
    if previous_response_id is not None:
        _mark_generated_input(context)
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


def _preemption_messages(corpus: Corpus, marker: str) -> list[dict[str, Any]]:
    messages = copy.deepcopy(corpus.shape_messages("interferer-256"))
    messages[0]["content"] = marker + "。" + messages[0]["content"]
    messages[-1]["content"] += (
        "\n每章写出具体场景、人物对话和连续动作。慢慢展开环境变化、人物记忆与线索发现，"
        "详细描写每个转折的过程。让故事自然推进，保持情节连贯。"
        "重点描写设备操作的细节，以及主角在不同阶段的观察、推理和具体行动。"
        "保持每章的完整性。"
    )
    return messages


class _LiveRequestLog:
    def __init__(self, context: CaseContext) -> None:
        value = context.notes.get("request_log_jsonl")
        self.path = Path(value) if isinstance(value, str) else None
        self.offset = self.path.stat().st_size if self.path is not None and self.path.is_file() else 0
        self.samples = 0
        self.error: str | None = None

    def poll(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.is_file() or self.error is not None:
            return []
        samples = []
        try:
            with self.path.open(encoding="utf-8") as source:
                source.seek(self.offset)
                while True:
                    line = source.readline()
                    if not line or not line.endswith("\n"):
                        break
                    self.offset = source.tell()
                    event = json.loads(line)
                    if isinstance(event, dict) and event.get("event") in {
                        "throughput", "request_scheduling",
                    }:
                        self.samples += 1
                        samples.append({"event": event, "observed_ns": time.perf_counter_ns()})
        except (OSError, UnicodeError, ValueError) as error:
            self.error = str(error)
        return samples


def _fixed_request_graph(
    context: CaseContext,
    requests: list[tuple[str, Any]],
    offsets_ns: tuple[int, ...],
    facts: dict[str, tuple[int | None, int]],
    on_sample: Callable[[dict[str, Any], list[RequestHandle]], None] | None = None,
    *,
    prepare_at_arrival: bool = False,
) -> list[RequestHandle]:
    handles = [] if prepare_at_arrival else [context.prepare(role, request) for role, request in requests]
    live_log = _LiveRequestLog(context) if on_sample is not None else None
    gates = [threading.Event() for _ in handles]
    for handle, gate in zip(handles, gates, strict=True):
        handle.start(gate)
    origin_ns = time.perf_counter_ns()
    arrivals = [
        {"role": role, "offset_ns": offset, "planned_ns": origin_ns + offset}
        for (role, _), offset in zip(requests, offsets_ns, strict=True)
    ]
    context.notes["arrival_mode"] = "fixed_schedule"
    context.notes["arrivals"] = arrivals
    for index, arrival in enumerate(arrivals):
        while (remaining_ns := arrival["planned_ns"] - time.perf_counter_ns()) > 0:
            time.sleep(remaining_ns / 1e9)
        if prepare_at_arrival:
            # A long arrival trace must not leave preconnected sockets idle on the server.
            # Preparation and connect still precede t0; their delay is visible as send lateness.
            role, request = requests[index]
            handle = context.prepare(role, request)
            handles.append(handle)
            arrival["released_ns"] = time.perf_counter_ns()
            handle.start()
            continue
        arrival["released_ns"] = time.perf_counter_ns()
        gates[index].set()
    try:
        if live_log is not None and live_log.path is not None:
            deadline = time.monotonic() + context.timeout
            while True:
                for sample in live_log.poll():
                    on_sample(sample, handles)
                if all(handle.is_done for handle in handles):
                    break
                if time.monotonic() >= deadline:
                    raise CaseExecutionError("fixed request graph did not terminate")
                time.sleep(0.01)
        context.wait_all(handles)
    finally:
        observed = []
        for handle, arrival in zip(handles, arrivals, strict=True):
            arrival["sent_ns"] = handle.sent_ns
            arrival["lateness_ns"] = (
                handle.sent_ns - arrival["planned_ns"] if handle.sent_ns is not None else None
            )
            usage = handle.as_record()["usage"]
            input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
            expected_input, output_limit = facts[handle.role]
            observed.append({
                "role": handle.role, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "nominal_input_matched": input_tokens == expected_input
                                         if input_tokens is not None and expected_input is not None else None,
                "output_limit_reached": output_tokens == output_limit if output_tokens is not None else None,
            })
        context.notes["observed_workload"] = observed
        if live_log is not None:
            context.notes["live_sampling"] = {
                "configured": live_log.path is not None, "samples_seen": live_log.samples,
                "error": live_log.error,
            }
    return handles


def _preemption_load(restore: str) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        # Offline token counts describe the fixture; serving usage remains authoritative.
        context.notes.update({
            "mechanism_requirements": ["preemption", restore],
            "workload_shape": {"requests": 2, "nominal_input_tokens": 192,
                               "max_output_tokens": 256, "eos_policy": "normal"},
        })
        handles = _fixed_request_graph(
            context,
            [(role, chat_request(context.model, _preemption_messages(corpus, marker), 256))
             for role, marker in (("a", "ALPHA"), ("b", "BRAVO"))],
            (0, 10_000_000), {"a": (192, 256), "b": (192, 256)},
        )
        _require_successes(context, handles)
    return run


def _shared_growth_messages(corpus: Corpus, branch: str) -> list[dict[str, Any]]:
    system = corpus.shape_messages("state-2k-a")[0]["content"]
    question = corpus.shape_messages("interferer-256")[-1]["content"]
    question += (
        "\n每章写出具体场景、人物对话和连续动作。慢慢展开环境变化、人物记忆与线索发现，"
        "详细描写每个转折的过程。让故事自然推进，保持情节连贯。"
        f"\n故事分支 {branch}。"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system,
                                        "prompt_cache_breakpoint": {"mode": "explicit"}}]},
        {"role": "user", "content": question},
    ]


def _shared_growth_recovery(context: CaseContext, corpus: Corpus) -> None:
    seed = context.start("seed", chat_request(context.model, _shared_growth_messages(corpus, "A"), 1))
    context.require_success(seed)
    context.notes.update({
        "mechanism_requirements": ["preemption", "replay_restore", "replay_with_other_progress"],
        "mechanism_observations": {
            "replay_with_other_progress": "unavailable", "recovery_arrival": "unavailable",
        },
        "throughput_comparable": False,
        "throughput_limitation": (
            "Two short requests arrive after an observed Replay restoration; their arrival time "
            "depends on engine recovery progress."
        ),
        "request_classes": {
            "seed": "setup", "a": "initial-shared-growth", "b": "initial-shared-growth",
            "c": "initial-shared-growth", "short-1": "initial-short", "short-2": "initial-short",
            "recovery-short-1": "recovery-arrival", "recovery-short-2": "recovery-arrival",
        },
        "workload_shape": {
            "shared_system_frontier_tokens": 2026,
            "shared_prefix_probe_roles": ["a", "b", "c"],
            "initial_requests": 5, "recovery_arrivals": 2,
            "long_input_tokens": 2176, "short_input_tokens": 30,
            "initial_output_limits": [128, 512, 512, 16, 16],
            "recovery_output_limit": 16, "eos_policy": "normal",
        },
    })

    recovery_arrivals: list[RequestHandle] = []

    def submit_after_replay(sample: dict[str, Any], _handles: list[RequestHandle]) -> None:
        event = sample["event"]
        if recovery_arrivals or event.get("event") != "request_scheduling":
            return
        context.notes["mechanism_observations"]["recovery_arrival"] = "not_observed"
        if event.get("transition") != "restored" or event.get("route") != "replay":
            return
        context.notes["recovery_arrival_trigger"] = {
            "event": event, "observed_ns": sample["observed_ns"],
        }
        for role in ("recovery-short-1", "recovery-short-2"):
            recovery_arrivals.append(context.start(
                role, chat_request(context.model, corpus.shape_messages("short-32"), 16)))
        context.notes["mechanism_observations"]["recovery_arrival"] = "observed"

    requests = [
        (role, chat_request(context.model, _shared_growth_messages(corpus, branch), limit))
        for role, branch, limit in (("a", "A", 128), ("b", "B", 512), ("c", "C", 512))
    ]
    requests.extend(
        (role, chat_request(context.model, corpus.shape_messages("short-32"), 16))
        for role in ("short-1", "short-2")
    )
    handles = _fixed_request_graph(
        context, requests, (0, 10_000_000, 20_000_000, 50_000_000, 100_000_000),
        {"a": (2176, 128), "b": (2176, 512), "c": (2176, 512),
         "short-1": (30, 16), "short-2": (30, 16)}, submit_after_replay,
    )
    context.notes["arrival_mode"] = "fixed_initial_then_recovery_event"
    _require_successes(context, [*handles, *recovery_arrivals])


def _vision_growth_replay(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.media_messages("image-chart")
    prompt = corpus.shape_messages("interferer-256")[-1]["content"] + (
        "\n详细讲述每一步的判断过程，逐段分析人物和环境的变化。保持故事连贯，不要提前结束。"
    )
    for part in messages[0]["content"]:
        if part.get("type") == "text":
            part["text"] = prompt
    context.notes.update({
        "mechanism_requirements": ["vision_preemption", "vision_replay"],
        "mechanism_role_requirements": {
            "vision_preemption": {"role": "vision", "counter": "preemptions"},
            "vision_replay": {"role": "vision", "counter": "replay_restores"},
        },
        "workload_shape": {"text_input_tokens": 192, "vision_input_tokens": 512,
                           "text_output_limit": 768, "vision_output_limit": 512,
                           "media_fixture": "image-chart", "eos_policy": "normal"},
    })
    handles = _fixed_request_graph(
        context,
        [("text", chat_request(context.model, _preemption_messages(corpus, "ALPHA"), 768)),
         ("vision", chat_request(context.model, messages, 512))],
        (0, 10_000_000), {"text": (192, 768), "vision": (512, 512)},
    )
    _require_successes(context, handles)


def _anonymous_hot(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.shape_messages("long-8k-16")
    source = context.start(
        "source",
        chat_request(context.model, source_messages, 16),
    )
    context.require_success(source)
    continuation_messages = [*source_messages, _assistant(context, source)]
    continuation_messages.append(
        {"role": "user", "content": "Now answer with the saved code only."}
    )
    continuation = context.start(
        "continuation",
        chat_request(context.model, continuation_messages, 32),
    )
    context.require_success(continuation)


def _state_probe(context: CaseContext, corpus: Corpus, label: str, turn: int) -> None:
    request = context.start(
        f"{label}{turn}", chat_request(context.model, corpus.state_messages(label, turn), 32)
    )
    context.require_success(request)


def _shared_state_working_set(context: CaseContext, corpus: Corpus) -> None:
    for turn in range(2):
        for label in "abc":
            _state_probe(context, corpus, label, turn)
    _state_probe(context, corpus, "a", 2)
    for turn in range(4):
        for label in "def":
            _state_probe(context, corpus, label, turn)
    _state_probe(context, corpus, "a", 3)


def _private_state_working_set(
    context: CaseContext, corpus: Corpus, *, automatic_shared: bool = True,
) -> None:
    context.notes["automatic_shared_capture"] = automatic_shared
    for labels, rounds in (("ab", 2), ("cd", 4)):
        histories = {label: corpus.state_messages(label) for label in labels}
        for turn in range(rounds):
            for label in labels:
                history = histories[label]
                if turn:
                    history.append({"role": "user", "content": "Continue with one more detail."})
                request = chat_request(context.model, history, 32)
                if not automatic_shared:
                    request.payload["prompt_cache_options"] = {"mode": "explicit"}
                response = context.start(f"{label}{turn}", request)
                context.require_success(response)
                history.append(_assistant(context, response))


def _shared_state_hot_prefix(context: CaseContext, corpus: Corpus) -> None:
    _state_probe(context, corpus, "a", 0)
    _state_probe(context, corpus, "a", 1)
    for turn, label in enumerate("bcdef", start=2):
        _state_probe(context, corpus, label, 0)
        _state_probe(context, corpus, "a", turn)


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
            previous_response_id=_response_id(context, source),
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
            previous_response_id=_response_id(context, a1),
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
            previous_response_id=_response_id(context, b1),
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
            previous_response_id=_response_id(context, a1),
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
            previous_response_id=_response_id(context, b1),
        ),
    )
    context.require_success(b2)


def _session_rotation_55k_host(context: CaseContext, corpus: Corpus) -> None:
    source_ids: list[str] = []
    for index in range(6):
        source = _responses_shape(
            context,
            corpus,
            f"rotation-55k-{index}",
            f"source-{index}",
            store=True,
        )
        context.require_success(source)
        source_ids.append(_response_id(context, source))

    def resume(role: str, source_index: int) -> None:
        request = context.start(
            role,
            responses_request(
                context.model,
                "Return the retained answer and its session marker in one short line.",
                32,
                store=True,
                previous_response_id=source_ids[source_index],
            ),
        )
        context.require_success(request)

    resume("warm-context-0", 0)
    for round_index in range(1, 4):
        for source_index in range(6):
            resume(f"round-{round_index}-context-{source_index}", source_index)


class _BackgroundResponseLoops:
    """Keep two store-free streamed generations active across a long foreground history."""

    def __init__(self, context: CaseContext, initial: list[RequestHandle]) -> None:
        self._context = context
        self._current = list(initial)
        self._generations = [0 for _ in initial]
        self._errors: list[str] = []
        self._cancelled_handles: set[RequestHandle] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(index,),
                name=f"ttft-background-loop-{index}",
            )
            for index in range(len(initial))
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _request(context: CaseContext, index: int) -> Any:
        return responses_request(
            context.model,
            (
                f"Background decode worker {index}: emit a long numbered sequence, one short "
                "item per line, until the output limit."
            ),
            900,
            store=False,
        )

    @classmethod
    def start(cls, context: CaseContext) -> _BackgroundResponseLoops:
        handles = context.barrier(
            (
                (f"background-{index}-generation-0", cls._request(context, index))
                for index in range(2)
            )
        )
        for handle in handles:
            handle.wait_first_output(context.timeout)
        return cls(context, handles)

    def _record_error(self, index: int, error: BaseException) -> None:
        with self._lock:
            self._errors.append(f"background loop {index}: {type(error).__name__}: {error}")
            self._stop.set()

    def _run(self, index: int) -> None:
        try:
            while True:
                with self._lock:
                    handle = self._current[index]
                handle.wait_done(self._context.timeout)
                record = handle.as_record()
                with self._lock:
                    owned_cancellation = handle in self._cancelled_handles
                clean_cancellation = (
                    owned_cancellation
                    and record["outcome"] == "cancelled"
                    and record["cancel_requested"]
                    and record["transport_cancelled"]
                    and (record["http_status"] is None or 200 <= record["http_status"] < 300)
                    and all(record[key] is None for key in (
                        "transport_error", "protocol_error", "error_code", "error_message",
                    ))
                    and not any(event["kind"] == "error" for event in record["events"])
                )
                if record["outcome"] != "success" and not clean_cancellation:
                    errors = {key: record[key] for key in (
                        "transport_error", "protocol_error", "error_code", "error_message",
                    ) if record[key] is not None}
                    raise CaseExecutionError(
                        f"{handle.role} ended before its replacement: {record['outcome']}, "
                        f"HTTP {record['http_status']}, errors={errors}"
                    )
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._generations[index] += 1
                    generation = self._generations[index]
                    handle = self._context.start(
                        f"background-{index}-generation-{generation}",
                        self._request(self._context, index),
                    )
                    self._current[index] = handle
                # require_active owns the foreground's first-output condition. The loop only
                # needs a terminal result before replacing this generation; stop can cancel it
                # before or after first output without turning that cancellation into a wait error.
        except BaseException as error:
            self._record_error(index, error)

    def require_active(self) -> None:
        deadline = time.monotonic() + self._context.timeout
        while True:
            with self._lock:
                errors = list(self._errors)
                current = list(self._current)
            if errors:
                raise CaseExecutionError(errors[0])
            if all(handle.first_output_ns is not None and not handle.is_done for handle in current):
                return
            if self._stop.is_set() or time.monotonic() >= deadline:
                detail = ", ".join(
                    f"{handle.role}:{handle.outcome()}" for handle in current
                )
                raise CaseExecutionError(
                    f"background streams were not simultaneously active: {detail}"
                )
            time.sleep(0.01)

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            for handle in self._current:
                if not handle.is_done and handle.cancel(only_if_unrequested=True) is not None:
                    self._cancelled_handles.add(handle)
        for thread in self._threads:
            thread.join(timeout=min(self._context.timeout, 5.0))
        live_threads = [thread.name for thread in self._threads if thread.is_alive()]
        with self._lock:
            errors = list(self._errors)
            generations = list(self._generations)
        self._context.notes["background_stream_generations"] = generations
        if live_threads:
            raise CaseExecutionError(
                f"background stream loops did not terminate: {', '.join(live_threads)}"
            )
        if errors:
            raise CaseExecutionError(errors[0])


def _second_rotation_messages(corpus: Corpus, index: int) -> list[dict[str, Any]]:
    facts = corpus.shape(f"rotation-55k-{index}")
    messages = corpus.shape_messages(f"rotation-55k-{index}")
    if not messages or messages[0].get("role") != "system":
        raise CaseExecutionError("55K rotation fixture has no leading system marker")
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise CaseExecutionError("55K rotation system marker is not text")
    original, separator, suffix = content.partition(" ")
    replacement = facts.get("second_cohort_label")
    if not isinstance(replacement, str):
        raise CaseExecutionError("55K rotation fixture has no second-cohort marker")
    if not separator or len(original) != len(replacement):
        raise CaseExecutionError("second 55K cohort marker does not preserve fixture shape")
    messages[0]["content"] = replacement + separator + suffix
    return messages


def _session_rotation_55k_two_cohort_stream(context: CaseContext, corpus: Corpus) -> None:
    # Reproduce the process history from #144: first establish the complete sequential rotation
    # estate, then keep two store-free streams live while a distinct second cohort is created and
    # resumed. Same-length leading marker replacements preserve the frozen 55K request shape while
    # making every second-cohort root diverge at the first system-content token.
    context.notes["arrival_mode"] = "causal_dynamic_background"
    context.notes["throughput_comparable"] = False
    context.notes["throughput_limitation"] = (
        "Background streams are replaced until the foreground graph finishes; submitted work "
        "depends on engine speed. Use this case for causal observations, not throughput comparison."
    )
    _session_rotation_55k_host(context, corpus)
    context.notes["workload_phases"] = {
        "first-cohort": [handle.role for handle in context.handles],
        "first-cohort-warm": [handle.role for handle in context.handles
                              if handle.role == "warm-context-0" or handle.role.startswith("round-")],
    }
    background = _BackgroundResponseLoops.start(context)
    try:
        source_ids: list[str] = []
        for index in range(6):
            background.require_active()
            facts = corpus.shape(f"rotation-55k-{index}")
            source = context.start(
                f"cohort-b-source-{index}",
                responses_request(
                    context.model,
                    _second_rotation_messages(corpus, index),
                    facts["max_output_tokens"],
                    store=True,
                ),
            )
            context.require_success(source)
            source_ids.append(_response_id(context, source))

        for round_index in range(1, 4):
            for source_index in range(6):
                background.require_active()
                request = context.start(
                    f"cohort-b-round-{round_index}-context-{source_index}",
                    responses_request(
                        context.model,
                        "Return the retained answer and its session marker in one short line.",
                        32,
                        store=True,
                        previous_response_id=source_ids[source_index],
                    ),
                )
                context.require_success(request)
    finally:
        background.stop()


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
            previous_response_id=_response_id(context, source),
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


def _unmarked_repeat_after_interference(context: CaseContext, corpus: Corpus) -> None:
    messages = corpus.shape_messages("unmarked-common-a")
    facts = corpus.shape("unmarked-common-a")
    first = context.start(
        "first-observation",
        anthropic_request(context.model, messages, facts["max_output_tokens"]),
    )
    context.require_success(first)
    second = context.start(
        "second-observation",
        anthropic_request(context.model, messages, facts["max_output_tokens"]),
    )
    context.require_success(second)
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
        "reuse",
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


def _shared_prefix_competition(context: CaseContext, corpus: Corpus) -> None:
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


def _shared_tool_prefix_competition(context: CaseContext, corpus: Corpus) -> None:
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


def _shared_tools_revisit(context: CaseContext, corpus: Corpus) -> None:
    for role, question, changed in (
        ("source", "Summarize what these tools can inspect.", False),
        ("stable", "Name one safe repository operation.", False),
        ("changed", "Name one safe repository operation.", True),
    ):
        request = context.start(
            role,
            anthropic_request(
                context.model,
                [{"role": "user", "content": f"Do not call a tool. {question}"}],
                32,
                tools=corpus.client_tools(changed_first=changed),
            ),
        )
        context.require_success(request)


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


def _finish_decode_overlap(
    context: CaseContext,
    corpus: Corpus,
    holder: RequestHandle,
    foreground: RequestHandle,
) -> None:
    context.require_success(foreground)
    completed_ns = foreground.completed_ns
    if not isinstance(completed_ns, int):
        raise CaseExecutionError(f"{foreground.role} completed without a timestamp")
    deadline = time.monotonic() + context.timeout
    while True:
        record = holder.as_record()
        resumed_ns = record["last_output_ns"]
        if isinstance(resumed_ns, int) and resumed_ns > completed_ns:
            break
        if holder.is_done:
            raise CaseExecutionError(
                f"{holder.role} ended before producing output after {foreground.role} completed"
            )
        if time.monotonic() >= deadline:
            raise CaseExecutionError(
                f"{holder.role} did not resume output after {foreground.role} completed"
            )
        time.sleep(0.005)
    context.notes["decode_overlap"] = {
        "foreground": foreground.role,
        "foreground_completed_ns": completed_ns,
        "holder_resumed_output_ns": resumed_ns,
    }
    _require_order(
        context,
        f"holder.first < {foreground.role}.sent < {foreground.role}.completed < holder.resumed",
        (
            ("holder.first", holder.first_output_ns),
            (f"{foreground.role}.sent", foreground.sent_ns),
            (f"{foreground.role}.completed", completed_ns),
            ("holder.resumed", resumed_ns),
        ),
    )
    cancel_ns = holder.cancel()
    holder.wait_done(context.timeout)
    _require_transport_cancellation(context, holder, cancel_ns)
    context.require_success(_chat_shape(context, corpus, "short-32", "cleanup-probe"))


def _decode_with_short_arrival(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    short = _chat_shape(context, corpus, "short-32", "short")
    _finish_decode_overlap(context, corpus, holder, short)


def _short_behind_long(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    head = _chat_shape(context, corpus, "long-8k-32", "head")
    head.wait_accepted(context.timeout)
    short = _chat_shape(context, corpus, "short-32", "short")
    _require_successes(context, (holder, head, short))
    _require_order(
        context,
        "head.accepted < short.sent < holder.completed",
        (
            ("head.accepted", head.accepted_ns),
            ("short.sent", short.sent_ns),
            ("holder.completed", holder.completed_ns),
        ),
    )


def _active_lanes_full(context: CaseContext, corpus: Corpus) -> None:
    context.notes["measurement_contract"] = "2"
    messages = corpus.shape_messages("holder-4096")
    holders = context.barrier(
        (
            (f"holder-{index}", chat_request(context.model, messages, 256))
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
        and probe.sent_ns < earliest_holder_completion,
        "probe.sent < min(holder.completed)",
        f"probe.sent={probe.sent_ns}, earliest_holder_completion={earliest_holder_completion}, probe.first={probe.first_output_ns}",
    )
    context.notes["lane_drain"] = {
        "holders": 8,
        "holder_output_limit": 256,
        "eos_policy": "normal",
        "earliest_holder_completion_ns": earliest_holder_completion,
        "last_holder_completion_ns": max(int(item.completed_ns) for item in holders),
        "probe_first_output_ns": probe.first_output_ns,
    }


def _session_publication_order(context: CaseContext, corpus: Corpus) -> None:
    parent = context.start(
        "parent",
        responses_request(context.model, "Create a session root.", 16, store=True),
    )
    context.require_success(parent)
    parent_id = _response_id(context, parent)
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
            previous_response_id=_response_id(context, newer),
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
    continuation_messages = [*source_messages, _assistant(context, seed)]
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
    history = [*source_messages, _assistant(context, source)]
    history.append({"role": "user", "content": "Recall the same image in one phrase."})
    continuation = context.start(
        "continuation", chat_request(context.model, history, 32)
    )
    context.require_success(continuation)


def _media_prefix_append(context: CaseContext, corpus: Corpus) -> None:
    source_messages = corpus.media_messages("image-chart")
    source = context.start("source", chat_request(context.model, source_messages, 32))
    context.require_success(source)
    history = [*source_messages, _assistant(context, source)]
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
    changed.append(_assistant(context, source))
    changed.append({"role": "user", "content": "Answer after the earlier image changed."})
    continuation = context.start(
        "changed-continuation", chat_request(context.model, changed, 32)
    )
    context.require_success(continuation)


def _media_preprocess_revisit(context: CaseContext, corpus: Corpus) -> None:
    for role, index in (("a-cold", 0), ("a-warm", 0), ("b", 1), ("c", 2), ("a-revisit", 0)):
        handle = context.start(
            role,
            chat_request(context.model, corpus.load_image_messages(index), 32),
        )
        context.require_success(handle)


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


def _media_during_decode(context: CaseContext, corpus: Corpus) -> None:
    holder = _chat_shape(context, corpus, "holder-4096", "holder")
    holder.wait_first_output(context.timeout)
    media = context.start(
        "media",
        chat_request(context.model, corpus.media_messages("many-image-28-a"), 32),
    )
    _finish_decode_overlap(context, corpus, holder, media)


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


def _resident_competition(order: str) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        decode_requests = [
            (f"decode-{label}", chat_request(context.model, _preemption_messages(corpus, marker), 512))
            for label, marker in (("a", "ALPHA"), ("b", "BRAVO"))
        ]
        context.notes.update({
            "request_classes": {"prefill": "long-prefill", "decode-a": "decode", "decode-b": "decode"},
            "expected_engine_order_groups": (
                [["decode-a", "decode-b"], ["prefill"]] if order == "decode-first"
                else [["prefill"], ["decode-a", "decode-b"]]
            ),
            "mechanism_requirements": ["expected_engine_order"],
            "workload_shape": {"prefill_input_tokens": 7680, "decode_input_tokens": 192,
                               "decode_output_limit": 512, "eos_policy": "normal"},
        })
        if order == "decode-first":
            decodes = context.barrier(decode_requests)
            for handle in decodes:
                handle.wait_first_output(context.timeout)
            prefill = _chat_shape(context, corpus, "long-8k-32", "prefill")
        else:
            prefill = _chat_shape(context, corpus, "long-8k-32", "prefill")
            prefill.wait_accepted(context.timeout)
            decodes = context.barrier(decode_requests)
        _require_successes(context, [prefill, *decodes])
        if order == "decode-first":
            for handle in decodes:
                _require_order(context, f"{handle.role}.first < prefill.sent < {handle.role}.completed", (
                    ("decode.first", handle.first_output_ns), ("prefill.sent", prefill.sent_ns),
                    ("decode.completed", handle.completed_ns),
                ))
        else:
            context.require(
                all(handle.sent_ns is not None and prefill.first_output_ns is not None
                    and handle.sent_ns < prefill.first_output_ns for handle in decodes),
                "decode requests sent before prefill first output",
                f"decode.sent={[handle.sent_ns for handle in decodes]}, prefill.first={prefill.first_output_ns}",
            )
    return run


def _mixed_arrival_trace(interval_ms: int) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        requests = []
        facts = {}
        classes = {}
        # One fixed ordered workload, with three offered arrival rates. Growth requests have
        # short narrative inputs; their output limits do not suppress normal EOS.
        for round_index in range(3):
            shapes = (
                ("growth", _preemption_messages(corpus, "ALPHA"), 192, 512),
                ("long-prefill", corpus.shape_messages("long-8k-32"), 7680, 32),
                ("short", corpus.shape_messages("short-32"), 30, 16),
                ("medium-prefill", corpus.state_messages("abcdef"[round_index]), 2048, 32),
            )
            for kind, messages, tokens, limit in shapes:
                role = f"round-{round_index}-{kind}"
                requests.append((role, chat_request(context.model, messages, limit)))
                facts[role] = (tokens, limit)
                classes[role] = kind
        context.notes.update({
            "request_classes": classes,
            "workload_shape": {"rounds": 3, "requests": len(requests),
                               "arrival_interval_ms": interval_ms, "eos_policy": "normal"},
        })
        handles = _fixed_request_graph(
            context, requests, tuple(index * interval_ms * 1_000_000 for index in range(len(requests))),
            facts, prepare_at_arrival=True,
        )
        _require_successes(context, handles)
    return run


def _agent_preempted_continuation(route: str) -> CaseFunction:
    def run(context: CaseContext, corpus: Corpus) -> None:
        history = _preemption_messages(corpus, "ALPHA")
        source = context.start("agent-source", chat_request(context.model, history, 32))
        context.require_success(source)
        history = [*history, _assistant(context, source),
                   {"role": "user", "content": "Continue the story with the next scene."}]
        turn = context.start("agent-turn", chat_request(context.model, history, 32))
        context.require_success(turn)
        history = [*history, _assistant(context, turn),
                   {"role": "developer", "content": "Keep the established names and facts unchanged."},
                   {"role": "user", "content": "Continue the story in detail, including the next investigation and dialogue."}]
        context.notes.update({
            "mechanism_requirements": ["agent_preemption", "agent_restore"],
            "mechanism_role_requirements": {
                "agent_preemption": {"role": "agent-growing", "counter": "preemptions"},
                "agent_restore": {"role": "agent-growing", "counter": f"{route}_restores"},
            },
            "request_classes": {
                "agent-source": "setup", "agent-turn": "setup", "contender": "growth",
                "agent-growing": "agent-under-pressure", "agent-retry": "agent-retry",
                "agent-followup": "agent-continuation",
            },
        })
        contender = context.start(
            "contender", chat_request(context.model, _preemption_messages(corpus, "BRAVO"), 512))
        contender.wait_first_output(context.timeout)
        agent = context.start("agent-growing", chat_request(context.model, history, 256))
        _require_successes(context, [contender, agent])
        retry = context.start("agent-retry", chat_request(context.model, history, 32))
        context.require_success(retry)
        followup = context.start("agent-followup", chat_request(
            context.model, [*history, _assistant(context, agent),
                            {"role": "user", "content": "Summarize the investigation in one short sentence."}], 32))
        context.require_success(followup)
    return run


def _host_history_pressure_cancel(context: CaseContext, corpus: Corpus) -> None:
    histories = [_preemption_messages(corpus, marker) for marker in ("OMEGA", "SIGMA")]
    for index, history in enumerate(histories):
        for step in ("seed", "warm"):
            context.require_success(context.start(
                f"history-{index}-{step}", chat_request(context.model, history, 16)))
    context.notes.update({
        "throughput_comparable": False,
        "throughput_limitation": "A request-specific paused event triggers cancellation; completed work is conditional.",
        "mechanism_requirements": ["preemption", "snapshot_restore", "pressure_cancellation"],
        "mechanism_observations": {"pressure_cancellation": "unavailable"},
        "workload_shape": {"histories": 2, "pressure_requests": 3, "pressure_input_tokens": 192,
                           "pressure_output_limit": 256, "eos_policy": "normal"},
    })

    def cancel_paused(sample: dict[str, Any], handles: list[RequestHandle]) -> None:
        event = sample["event"]
        if event.get("event") != "request_scheduling" or "pressure_cancel" in context.notes:
            return
        context.notes["mechanism_observations"]["pressure_cancellation"] = "not_observed"
        if event.get("transition") != "paused":
            return
        wire = event.get("request", {}).get("http_request_id")
        if not isinstance(wire, str) or not wire:
            return
        target = next((handle for handle in handles
                       if handle.as_record().get("wire_request_id") == wire), None)
        if target is None or target.is_done:
            return
        context.notes["pressure_cancel"] = {
            "target_role": target.role, "observed_transition": "paused", "event": event,
            "cancel_ns": target.cancel(),
        }

    handles = _fixed_request_graph(
        context,
        [(role, chat_request(context.model, _preemption_messages(corpus, marker), 256))
         for role, marker in (("a", "ALPHA"), ("b", "BRAVO"), ("c", "DELTA"))],
        (0, 10_000_000, 20_000_000), {role: (192, 256) for role in ("a", "b", "c")}, cancel_paused,
    )
    cancellation = context.notes.get("pressure_cancel")
    for handle in handles:
        if cancellation is not None and cancellation["target_role"] == handle.role:
            cancellation["outcome"] = handle.outcome()
            context.notes["mechanism_observations"]["pressure_cancellation"] = (
                "observed" if handle.outcome() == "cancelled" else "not_observed")
            if handle.outcome() not in {"cancelled", "success"}:
                context.require(False, "pressure cancellation terminated normally", handle.outcome(),
                                dimension="request_outcome")
        else:
            context.require_success(handle)
    for index, history in enumerate(histories):
        context.require_success(context.start(
            f"history-{index}-probe", chat_request(context.model, history, 16)))


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
    _definition(
        "shared-growth-recovery", "openai_chat", "shared-growth-recovery", "scheduling",
        ("state-2k-a", "interferer-256", "short-32"),
        "Three growing shared-prefix branches and two fixed short arrivals, then two new short requests after Replay restoration.",
        _shared_growth_recovery,
    ),
    _definition(
        "host-history-pressure-cancel", "openai_chat", "host-history-pressure", "scheduling",
        ("interferer-256",),
        "Two retained histories compete with three growing requests in limited Host storage; cancel a specifically observed paused request.",
        _host_history_pressure_cancel,
    ),
    _definition(
        "vision-growth-replay", "openai_chat", "vision-growth-replay", "scheduling",
        ("interferer-256", "image-chart"),
        "Fixed Text/Vision growth with Host disabled; require the Vision request's own replay.",
        _vision_growth_replay,
    ),
    _definition(
        "preemption-replay", "openai_chat", "preemption-replay", "scheduling",
        ("interferer-256",),
        "Two fixed arrivals compete for 512 KV tokens with no Host snapshot capacity; "
        "observe preemption and replay separately from request completion.",
        _preemption_load("replay_restore"),
    ),
    _definition(
        "preemption-snapshot", "openai_chat", "preemption-snapshot", "scheduling",
        ("interferer-256",),
        "The same two fixed arrivals with Host snapshot capacity; observe preemption and "
        "snapshot restoration separately from request completion.",
        _preemption_load("snapshot_restore"),
    ),
    _definition(
        "shared-state-working-set-shift", "openai_chat", "cache-state-working-set", "resource",
        tuple(f"state-2k-{label}" for label in "abcdef"),
        "Two three-conversation working sets in one process under State pressure, followed by repeated probes.",
        _shared_state_working_set,
    ),
    _definition(
        "private-state-working-set-shift", "openai_chat", "cache-private-working-set", "resource",
        tuple(f"state-2k-{label}" for label in "abcd"),
        "Two pairs of full-history conversations compete for private checkpoint State capacity.",
        _private_state_working_set,
    ),
    _definition(
        "private-only-working-set-shift", "openai_chat", "cache-private-working-set", "resource",
        tuple(f"state-2k-{label}" for label in "abcd"),
        "The same full-history conversation working sets with automatic shared capture explicitly disabled.",
        lambda context, corpus: _private_state_working_set(context, corpus, automatic_shared=False),
    ),
    _definition(
        "shared-state-hot-prefix", "openai_chat", "cache-state-working-set", "resource",
        tuple(f"state-2k-{label}" for label in "abcdef"),
        "A frequently revisited prefix is interleaved with five one-use conversations.",
        _shared_state_hot_prefix,
    ),
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
    _definition("session-rotation-55k-host", "openai_responses", "cache-rotation-55k-host", "resource", tuple(f"rotation-55k-{index}" for index in range(6)), "Six early-divergent 55K Responses roots, one warm branch, then three sequential round-robin branch rounds under Device/Host KV pressure.", _session_rotation_55k_host),
    _definition(
        "session-rotation-55k-two-cohort-stream",
        "openai_responses",
        "cache-rotation-55k-host",
        "resource",
        tuple(f"rotation-55k-{index}" for index in range(6)),
        "A complete first 55K rotation estate followed by two continuous store-free streams, "
        "six distinct second-cohort roots, and three second-cohort resume rounds under saturated "
        "Host State descriptors.",
        _session_rotation_55k_two_cohort_stream,
        symmetric_role_groups=tuple(
            SymmetricRoleGroup(
                f"cohort-b-round-{round_index}",
                tuple(
                    f"cohort-b-round-{round_index}-context-{source_index}"
                    for source_index in range(6)
                ),
            )
            for round_index in range(1, 4)
        ),
    ),
    _definition("unmarked-common-prefix-miss", "openai_chat", "cache-hot", "private", ("unmarked-common-a", "unmarked-common-b"), "Raw token commonality without a checkpoint.", _unmarked_common),
    _definition("resume-after-interference-device", "openai_responses", "cache-pressure-device", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Device-resident source.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-state-host", "openai_responses", "cache-pressure-state-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Host State restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-kv-host", "openai_responses", "cache-pressure-kv-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with Host KV restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-both-host", "openai_responses", "cache-pressure-both-host", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with State and KV restore.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("resume-after-interference-evicted", "openai_responses", "cache-pressure-evict", "resource", ("long-8k-16", "interferer-256"), "Pressure graph with no legal retained replica.", _pressure_graph, symmetric_role_groups=(SymmetricRoleGroup("interferers", ("interferer-b", "interferer-c")),)),
    _definition("continuation-cache-off", "openai_responses", "cache-off", "control", ("long-8k-16",), "Full-history Responses control with Engine reuse disabled.", _cache_off),
    _definition("shared-sequential", "anthropic_messages", "shared-prefix", "shared", ("system-a",), "Marked Anthropic shared prefix sequential reuse.", _shared_sequential),
    _definition("shared-openai-explicit", "openai_chat", "shared-prefix", "shared", ("system-a",), "OpenAI explicit system boundary shared by different suffixes.", _openai_explicit_shared),
    _definition("shared-openai-implicit", "openai_chat", "prefix-competition", "shared", ("unmarked-common-a",), "OpenAI automatic-prefix reuse after an independent request.", _openai_implicit_shared),
    _definition("unmarked-repeat-after-interference", "anthropic_messages", "prefix-competition", "private", ("unmarked-common-a",), "Repeat an unmarked prompt before and after an independent request.", _unmarked_repeat_after_interference),
    _definition("shared-fanout", "anthropic_messages", "shared-prefix", "shared", ("system-a",), "Concurrent branches from a non-aligned shared prefix.", _shared_fanout, symmetric_role_groups=(SymmetricRoleGroup("branches", ("branch-b", "branch-c")),)),
    _definition("shared-prefix-competition", "anthropic_messages", "prefix-competition", "shared", ("system-a", "system-b"), "Revisit a marked prefix after a competing prefix and an independent request.", _shared_prefix_competition),
    _definition("shared-tool-prefix-competition", "anthropic_messages", "prefix-competition", "shared", ("system-a", "client-tools-32"), "A marked system prefix and tool prefix compete for Device storage.", _shared_tool_prefix_competition),
    _definition("shared-tools-revisit", "anthropic_messages", "shared-prefix", "shared", ("client-tools-32",), "Stable 32-tool prefix reuse followed by an early tool identity change.", _shared_tools_revisit),
    _definition("short-during-prefill-128", "openai_chat", "scheduler-prefill-128", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 128-token chunks.", _short_during_prefill),
    _definition("short-during-prefill-1024", "openai_chat", "scheduler-overlap", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 1024-token chunks.", _short_during_prefill),
    _definition("short-during-prefill-4096", "openai_chat", "scheduler-prefill-4096", "scheduling", ("long-8k-32", "short-32"), "Arrival during prefill with 4096-token chunks.", _short_during_prefill),
    _definition("decode-with-short-arrival", "openai_chat", "scheduler-overlap", "scheduling", ("holder-4096", "short-32"), "Short arrival during decode; observe resumed holder output, cancel, then probe cleanup.", _decode_with_short_arrival),
    _definition("short-behind-long", "openai_chat", "scheduler-kv-pressure", "scheduling", ("holder-4096", "long-8k-32", "short-32"), "Short arrival behind a long request while a holder is decoding.", _short_behind_long),
    _definition("active-lanes-full-8", "openai_chat", "lane-limit-8", "scheduling", ("holder-4096", "short-32"), "Eight 256-output-limit holders, a queued short arrival, and natural lane release and drain.", _active_lanes_full, symmetric_role_groups=(SymmetricRoleGroup("holders", tuple(f"holder-{index}" for index in range(8))),)),
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
    _definition("media-preprocess-revisit", "openai_chat", "media-cache-tight", "media", ("load-image-00", "load-image-01", "load-image-02"), "A-A-B-C-A media preparation: cold, warm, then revisit under cache pressure with context reuse off.", _media_preprocess_revisit),
    _definition("many-image-28", "openai_chat", "vision-cache", "media", ("many-image-28-a",), "Legal high Vision and many-item request.", _many_image),
    _definition("many-image-28-thread-1", "openai_chat", "vision-thread-1", "media", ("many-image-28-a",), "Single preprocessing worker control.", _many_image),
    _definition("text-during-media-prepare", "openai_chat", "vision-concurrent", "media", ("many-image-28-a", "short-32"), "Text arrival while a complete media body is preparing.", _text_during_media_prepare),
    _definition("media-during-decode", "openai_chat", "vision-concurrent", "media", ("holder-4096", "many-image-28-a"), "Heavy media arrival during decode; observe resumed holder output, cancel, then probe cleanup.", _media_during_decode),
    _definition("two-heavy-media-arrivals", "openai_chat", "vision-concurrent", "media", ("many-image-28-a", "many-image-28-b"), "Two byte-distinct legal high-media arrivals.", _two_heavy_media, symmetric_role_groups=(SymmetricRoleGroup("media", ("media-a", "media-b")),)),
    _definition("vision-disabled", "openai_chat", "text-cold-8k", "rejection", ("image-chart",), "Media on a text-only Serve is rejected.", _vision_disabled),
    _definition("vision-envelope-over", "openai_chat", "vision-boundary", "rejection", ("many-image-33",), "Aggregate Vision envelope rejection.", _vision_envelope_over),
)



_DEFINITIONS += tuple(
    _definition(
        f"resident-{order}-{capacity}", "openai_chat", f"resident-{capacity}", "scheduling",
        ("long-8k-32", "interferer-256"),
        f"Resident Decode/Prefill competition with {order} engine ticket groups and {capacity} KV capacity.",
        _resident_competition(order),
        symmetric_role_groups=(SymmetricRoleGroup("decodes", ("decode-a", "decode-b")),),
    )
    for order in ("decode-first", "prefill-first") for capacity in ("roomy", "pressure")
) + tuple(
    _definition(
        f"mixed-arrivals-{label}", "openai_chat", "mixed-arrivals", "scheduling",
        ("short-32", "long-8k-32", "state-2k-a", "state-2k-b", "state-2k-c", "interferer-256"),
        f"Twelve finite mixed arrivals every {interval}ms, followed by natural completion and drain.",
        _mixed_arrival_trace(interval),
    )
    for label, interval in (("sparse", 2000), ("steady", 250), ("burst", 10))
) + tuple(
    _definition(
        f"agent-continuation-{route}", "openai_chat", f"agent-{route}", "scheduling",
        ("interferer-256",),
        f"A multi-turn agent experiences {route} recovery, then retries and continues its history.",
        _agent_preempted_continuation(route),
    ) for route in ("replay", "snapshot")
) + tuple(
    _definition(
        f"preemption-snapshot-{backend}", "openai_chat", f"preemption-snapshot-{backend}", "scheduling",
        ("interferer-256",), f"The minimal snapshot graph with {backend} speculative execution.",
        _preemption_load("snapshot_restore"),
    ) for backend in ("mtp", "dflash2")
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

    context.notes.setdefault("input_dependency", "fixed")
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
    constructed = not any(failure.dimension == "construction" for failure in context.failures)
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
        "status": (
            "request_failure" if constructed and context.failures
            else "constructed" if constructed else "not_constructed"
        ),
        "constructed": constructed,
        "request_checks_passed": not any(
            failure.dimension == "request_outcome" for failure in context.failures
        ),
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
