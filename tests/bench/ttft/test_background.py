from __future__ import annotations

from collections import deque
import threading
import time
from types import SimpleNamespace

import pytest

from tools.bench.ttft import cases
from tools.bench.ttft.cases import _BackgroundResponseLoops
from tools.bench.ttft.execution import CaseContext, CaseExecutionError, RequestHandle
from tools.ninfer_serve.client import (
    PreparedServeExchange,
    ProtocolEvent,
    ProtocolRequest,
    ServeExchangeResult,
)
from tools.streaming_http.client import HttpExchangeResult, HttpResponseHead


class ControlledExchange:
    """A stream whose completion and cancellation are controlled by the test's threads."""

    body_bytes = 0
    body = b"{}"

    def __init__(self, *, first_output=True, failure=None, complete_on_start=False):
        self.request = ProtocolRequest("openai_responses", "/test", {})
        self.first_output = first_output
        self.failure = failure
        self.cancel_ns = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.allow_return = threading.Event()
        self.allow_return.set()
        self.completed_normally = False
        self.complete_on_start = complete_on_start
        self.events = []
        self.on_event = None

    def cancel(self):
        self.cancel_ns = time.perf_counter_ns()
        self.finished.set()
        return self.cancel_ns

    def complete(self):
        self.completed_normally = True
        self.finished.set()

    def emit_output(self):
        event = ProtocolEvent("model_output", "delta", time.perf_counter_ns(), output="x")
        self.events.append(event)
        self.on_event(event)
        return event.received_ns

    def execute(self, *, on_sent, on_body_sent, on_event, on_headers=None):
        sent_ns = time.perf_counter_ns()
        on_sent(sent_ns)
        on_body_sent(sent_ns)
        events = self.events
        self.on_event = on_event
        if self.first_output:
            events.append(ProtocolEvent("model_output", "delta", sent_ns, output="x"))
            on_event(events[-1])
        self.started.set()
        if self.complete_on_start:
            self.complete()
        assert self.finished.wait(5), "test did not terminate its stream"
        assert self.allow_return.wait(5), "test did not release its completed stream"
        ended_ns = time.perf_counter_ns()
        if self.completed_normally:
            events.append(ProtocolEvent("terminal", "done", ended_ns))
            on_event(events[-1])
        if self.failure == "stream":
            events.append(ProtocolEvent("error", "error", ended_ns))
            on_event(events[-1])
        cancelled = self.cancel_ns is not None and not self.completed_normally
        return ServeExchangeResult(
            protocol=self.request.protocol,
            body_bytes=0,
            http=HttpExchangeResult(
                sent_ns=sent_ns, body_sent_ns=sent_ns, ended_ns=ended_ns,
                status=503 if self.failure == "http" else 200,
                error="socket timeout" if self.failure == "transport" else None,
                cancelled=cancelled, cancel_requested=self.cancel_ns is not None,
                cancel_ns=self.cancel_ns,
            ),
            events=events,
            protocol_error="invalid SSE" if self.failure == "protocol" else None,
        )


class ControlledClient:
    def __init__(self, exchanges):
        self.exchanges = deque(exchanges)
        self.prepare_entered = threading.Event()
        self.allow_prepare = threading.Event()
        self.allow_prepare.set()

    def prepare(self, request):
        self.prepare_entered.set()
        assert self.allow_prepare.wait(5), "test did not release request preparation"
        return self.exchanges.popleft()


def start_loop(exchanges, *, timeout=2, on_progress=None):
    client = ControlledClient(exchanges)
    context = CaseContext(client, "test-model", timeout, on_progress)
    initial = context.start("background-0-generation-0", exchanges[0].request)
    assert exchanges[0].started.wait(2)
    return context, client, _BackgroundResponseLoops(context, [initial])


@pytest.mark.parametrize("first_output", [False, True])
def test_stop_keeps_cancelled_replacement_in_raw_results(first_output):
    initial = ControlledExchange()
    replacement = ControlledExchange(first_output=first_output)
    context, _, loops = start_loop([initial, replacement])
    try:
        initial.complete()
        assert replacement.started.wait(2)
    finally:
        loops.stop()

    records = context.records()
    assert [record["outcome"] for record in records] == ["success", "cancelled"]
    assert (records[1]["first_output_ns"] is not None) == first_output
    assert records[1]["cancel_requested"] and records[1]["transport_cancelled"]
    assert context.notes["background_stream_generations"] == [1]


def test_stop_serializes_with_replacement_registration():
    initial = ControlledExchange()
    replacement = ControlledExchange(first_output=False)
    context, client, loops = start_loop([initial, replacement])
    client.prepare_entered.clear()
    client.allow_prepare.clear()
    initial.complete()
    assert client.prepare_entered.wait(2)
    errors = []
    stopping = threading.Event()

    def stop():
        stopping.set()
        try:
            loops.stop()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert stopping.wait(2)
    finally:
        client.allow_prepare.set()
        thread.join(3)

    assert not thread.is_alive()
    assert not errors
    assert [record["outcome"] for record in context.records()] == ["success", "cancelled"]


@pytest.mark.parametrize("failure", ["http", "transport", "protocol", "stream"])
def test_stop_does_not_hide_request_failure(failure):
    exchange = ControlledExchange(failure=failure)
    context, _, loops = start_loop([exchange])
    with pytest.raises(CaseExecutionError, match="ended before its replacement"):
        loops.stop()
    record = context.records()[0]
    assert record["http_status"] == (503 if failure == "http" else 200)
    if failure == "transport":
        assert record["transport_error"] == "socket timeout"
    elif failure == "protocol":
        assert record["protocol_error"] == "invalid SSE"
    elif failure == "stream":
        assert any(event["kind"] == "error" for event in record["events"])


def test_stop_does_not_hide_worker_wait_timeout():
    timed_out = threading.Event()

    def progress(stage, timestamp, fields):
        if stage == "request.wait_failed" and fields.get("target") == "worker_done":
            timed_out.set()

    exchange = ControlledExchange()
    context, _, loops = start_loop([exchange], timeout=0.05, on_progress=progress)
    try:
        assert timed_out.wait(2)
    finally:
        with pytest.raises(CaseExecutionError, match="did not terminate"):
            loops.stop()
        context.wait_all()
    assert context.records()[0]["outcome"] == "cancelled"


def test_cancellation_outside_loop_shutdown_remains_a_failure(monkeypatch):
    exchange = ControlledExchange()
    context, _, loops = start_loop([exchange])
    exchange.allow_return.clear()
    handle = context.handles[0]
    handle.cancel()
    original_cancel = handle.cancel
    stopping = threading.Event()
    errors = []

    def cancel(**kwargs):
        result = original_cancel(**kwargs)
        stopping.set()
        return result

    def stop():
        try:
            loops.stop()
        except CaseExecutionError as error:
            errors.append(str(error))

    monkeypatch.setattr(handle, "cancel", cancel)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert stopping.wait(2)
    finally:
        exchange.allow_return.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and "ended before its replacement" in errors[0]
    assert context.records()[0]["outcome"] == "cancelled"


def test_decode_overlap_waits_for_resumed_output_before_cancelling_and_probing(monkeypatch):
    holder = ControlledExchange()
    foreground = ControlledExchange(complete_on_start=True)
    probe = ControlledExchange(complete_on_start=True)
    context = CaseContext(ControlledClient([holder, foreground, probe]), "test-model", 2)
    resumed = []

    def resume_when_waiting(_seconds):
        assert foreground.finished.is_set()
        assert holder.cancel_ns is None
        resumed.append(holder.emit_output())

    monkeypatch.setattr(cases, "time", SimpleNamespace(
        monotonic=time.monotonic, sleep=resume_when_waiting,
    ))
    corpus = SimpleNamespace(
        shape=lambda _name: {"max_output_tokens": 32},
        shape_messages=lambda name: [{"role": "user", "content": name}],
    )
    try:
        cases.get_case("decode-with-short-arrival").run(context, corpus)
    finally:
        context.cancel_live()

    assert not context.failures
    records = {record["role"]: record for record in context.records()}
    assert records["holder"]["outcome"] == "cancelled"
    assert records["short"]["outcome"] == records["cleanup-probe"]["outcome"] == "success"
    assert records["short"]["completed_ns"] < resumed[0] <= records["holder"]["cancel_ns"]
    assert records["holder"]["completed_ns"] < records["cleanup-probe"]["sent_ns"]


def test_decode_overlap_does_not_hide_holder_finishing_before_resumption():
    holder = ControlledExchange(complete_on_start=True)
    foreground = ControlledExchange(complete_on_start=True)
    context = CaseContext(ControlledClient([holder, foreground]), "test-model", 2)
    corpus = SimpleNamespace(
        shape=lambda _name: {"max_output_tokens": 32},
        shape_messages=lambda name: [{"role": "user", "content": name}],
    )
    with pytest.raises(CaseExecutionError, match="ended before producing output"):
        cases.get_case("decode-with-short-arrival").run(context, corpus)
    assert holder.cancel_ns is None
    assert all(record["outcome"] == "success" for record in context.records())


def test_response_header_identity_is_available_while_the_stream_is_live():
    class PendingHttpExchange:
        cancel_ns = None

        def __init__(self):
            self.headers_sent = threading.Event()
            self.finish = threading.Event()

        def cancel(self):
            self.cancel_ns = time.perf_counter_ns()
            self.finish.set()
            return self.cancel_ns

        def execute(self, *, on_sent, on_body_sent, on_headers, on_chunk):
            sent = time.perf_counter_ns()
            headers = {"content-type": "text/event-stream", "x-request-id": "req-live"}
            on_sent(sent)
            on_body_sent(sent)
            on_headers(HttpResponseHead(200, "OK", headers, time.perf_counter_ns()))
            self.headers_sent.set()
            assert self.finish.wait(2)
            return HttpExchangeResult(
                sent_ns=sent, body_sent_ns=sent, ended_ns=time.perf_counter_ns(),
                status=200, headers=headers, cancel_requested=True, cancel_ns=self.cancel_ns,
            )

    raw = PendingHttpExchange()
    request = ProtocolRequest("openai_chat", "/v1/chat/completions", {})
    handle = RequestHandle("live", 0, PreparedServeExchange(request, b"{}", raw), None)
    handle.start()
    try:
        assert raw.headers_sent.wait(2)
        assert handle.result is None
        assert not handle.is_done
        assert handle.as_record()["wire_request_id"] == "req-live"
    finally:
        handle.cancel()
        handle.wait_done(2)
    assert handle.outcome() == "cancelled"
    assert handle.as_record()["wire_request_id"] == "req-live"
