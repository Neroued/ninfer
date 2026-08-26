"""Request workers and externally observable case-event coordination."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from tools.ninfer_serve.client import (
    NInferServeClient,
    PreparedServeExchange,
    ProtocolEvent,
    ProtocolRequest,
    ServeExchangeResult,
)


class CaseExecutionError(RuntimeError):
    pass


ProgressCallback = Callable[[str, int, dict[str, Any]], None]


@dataclass(frozen=True)
class FailedCondition:
    expression: str
    detail: str

    def as_json(self) -> dict[str, str]:
        return {"expression": self.expression, "detail": self.detail}


class RequestHandle:
    def __init__(
        self,
        role: str,
        order: int,
        prepared: PreparedServeExchange,
        on_progress: ProgressCallback | None,
    ) -> None:
        self.role = role
        self.order = order
        self.protocol = prepared.request.protocol
        self.stream = prepared.request.stream
        self._prepared = prepared
        self._on_progress = on_progress
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: ServeExchangeResult | None = None
        self._thread_error: str | None = None
        self._event_trace: list[dict[str, Any]] = []
        self._output_parts: list[str] = []

        self.sent_ns: int | None = None
        self.body_sent_ns: int | None = None
        self.accepted_ns: int | None = None
        self.first_output_ns: int | None = None
        self.completed_ns: int | None = None
        self.cancel_ns: int | None = None
        self.response_id: str | None = None

        self._sent = threading.Event()
        self._body_sent = threading.Event()
        self._accepted = threading.Event()
        self._first_output = threading.Event()
        self._completed = threading.Event()
        self._done = threading.Event()

    def _progress(
        self,
        stage: str,
        *,
        occurred_ns: int | None = None,
        **fields: Any,
    ) -> None:
        if self._on_progress is None:
            return
        self._on_progress(
            stage,
            occurred_ns if occurred_ns is not None else time.perf_counter_ns(),
            {"role": self.role, "order": self.order, **fields},
        )

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def output_text(self) -> str:
        with self._lock:
            return "".join(self._output_parts)

    @property
    def result(self) -> ServeExchangeResult | None:
        with self._lock:
            return self._result

    def _on_sent(self, timestamp: int) -> None:
        with self._lock:
            self.sent_ns = timestamp
        self._sent.set()

    def _on_body_sent(self, timestamp: int) -> None:
        with self._lock:
            self.body_sent_ns = timestamp
            sent_ns = self.sent_ns
        self._body_sent.set()
        # Do not report from _on_sent: it runs immediately before the first socket write and any
        # formatting or stderr I/O there would become part of TTFT. Both timestamps are published
        # only after the complete request body has left the client.
        if sent_ns is not None:
            self._progress("request.sent", occurred_ns=sent_ns)
        self._progress(
            "request.body_sent",
            occurred_ns=timestamp,
            upload_ms=(timestamp - sent_ns) / 1e6 if sent_ns is not None else None,
            body_bytes=self._prepared.body_bytes,
        )

    def _on_event(self, event: ProtocolEvent) -> None:
        with self._lock:
            first_accepted = event.kind == "accepted" and self.accepted_ns is None
            first_output = event.kind == "model_output" and self.first_output_ns is None
            first_terminal = event.kind in {"terminal", "error"} and self.completed_ns is None
            if first_accepted or first_output or first_terminal:
                self._event_trace.append(
                    {
                        "kind": event.kind,
                        "event_type": event.event_type,
                        "received_ns": event.received_ns,
                        "response_id": event.response_id,
                        "output_bytes": len(event.output.encode("utf-8")),
                        "error_code": event.error_code,
                    }
                )
            if event.response_id:
                self.response_id = event.response_id
            if event.kind == "accepted" and self.accepted_ns is None:
                self.accepted_ns = event.received_ns
                self._accepted.set()
            elif event.kind == "model_output":
                self._output_parts.append(event.output)
                if self.first_output_ns is None:
                    self.first_output_ns = event.received_ns
                    self._first_output.set()
            elif event.kind in {"terminal", "error"} and self.completed_ns is None:
                self.completed_ns = event.received_ns
                self._completed.set()
            sent_ns = self.sent_ns

        if first_accepted:
            self._progress(
                "request.accepted",
                occurred_ns=event.received_ns,
                event_type=event.event_type,
                since_sent_ms=(event.received_ns - sent_ns) / 1e6 if sent_ns is not None else None,
                response_id=event.response_id,
            )
        if first_output:
            self._progress(
                "request.first_output",
                occurred_ns=event.received_ns,
                event_type=event.event_type,
                ttft_ms=(event.received_ns - sent_ns) / 1e6 if sent_ns is not None else None,
                output_bytes=len(event.output.encode("utf-8")),
            )
        if first_terminal:
            self._progress(
                "request.terminal",
                occurred_ns=event.received_ns,
                event_type=event.event_type,
                elapsed_ms=(event.received_ns - sent_ns) / 1e6 if sent_ns is not None else None,
                error_code=event.error_code,
            )

    def _run(self, gate: threading.Event | None) -> None:
        if gate is not None:
            self._progress("request.gate_wait")
            gate.wait()
            self._progress("request.gate_released")
        try:
            result = self._prepared.execute(
                on_sent=self._on_sent,
                on_body_sent=self._on_body_sent,
                on_event=self._on_event,
            )
            with self._lock:
                self._result = result
                if self.sent_ns is None:
                    self.sent_ns = result.http.sent_ns
                if self.body_sent_ns is None:
                    self.body_sent_ns = result.http.body_sent_ns
                if self.completed_ns is None and (
                    not self.stream
                    or result.http.cancelled
                    or result.http.error is not None
                    or result.http.status is None
                    or not (200 <= result.http.status < 300)
                    or result.protocol_error is not None
                ):
                    self.completed_ns = result.http.ended_ns
                if result.http.cancelled:
                    self.cancel_ns = result.http.cancel_ns or self._prepared.cancel_ns
        except Exception as error:  # preserve worker failures as benchmark data
            message = f"{type(error).__name__}: {error}"
            with self._lock:
                self._thread_error = message
                if self.completed_ns is None:
                    self.completed_ns = time.perf_counter_ns()
            self._progress("request.error", error=message)
        finally:
            if self.sent_ns is not None:
                self._sent.set()
            if self.body_sent_ns is not None:
                self._body_sent.set()
            if self.completed_ns is not None:
                self._completed.set()
            self._done.set()
            record = self.as_record()
            self._progress(
                "request.finished",
                occurred_ns=record["completed_ns"],
                outcome=record["outcome"],
                http_status=record["http_status"],
                ttft_ms=(
                    record["ttft_ns"] / 1e6
                    if isinstance(record["ttft_ns"], int)
                    else None
                ),
                output_bytes=record["output_bytes"],
                error_code=record["error_code"],
            )

    def start(self, gate: threading.Event | None = None) -> None:
        with self._lock:
            if self._thread is not None:
                raise CaseExecutionError(f"request {self.role} was started twice")
            self._thread = threading.Thread(
                target=self._run,
                args=(gate,),
                name=f"ttft-{self.order}-{self.role}",
            )
            self._progress("request.started", gated=gate is not None)
            self._thread.start()

    def cancel(self) -> int:
        timestamp = self._prepared.cancel()
        with self._lock:
            if self.cancel_ns is None:
                self.cancel_ns = timestamp
            never_started = self._thread is None
            if never_started and self.completed_ns is None:
                self.completed_ns = timestamp
        if never_started:
            self._completed.set()
            self._done.set()
        self._progress(
            "request.cancelled",
            occurred_ns=timestamp,
            before_start=never_started,
        )
        if never_started:
            self._progress(
                "request.finished",
                occurred_ns=timestamp,
                outcome="cancelled",
                http_status=None,
                ttft_ms=None,
                output_bytes=0,
                error_code=None,
            )
        return timestamp

    def _wait_for(self, event: threading.Event, field: str, timeout: float) -> int:
        target = field.removesuffix("_ns")
        report_wait = target not in {"sent", "body_sent"}
        wait_started_ns = time.perf_counter_ns()
        if report_wait and not event.is_set():
            self._progress("request.wait", target=target, timeout_s=timeout)
        deadline = time.monotonic() + timeout
        while not event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._progress(
                    "request.wait_failed",
                    target=target,
                    reason="timeout",
                    waited_ms=(time.perf_counter_ns() - wait_started_ns) / 1e6,
                )
                raise CaseExecutionError(f"request {self.role} timed out waiting for {field}")
            event.wait(min(remaining, 0.05))
            if self._done.is_set() and not event.is_set():
                self._progress(
                    "request.wait_failed",
                    target=target,
                    reason=f"ended:{self.outcome()}",
                    waited_ms=(time.perf_counter_ns() - wait_started_ns) / 1e6,
                )
                raise CaseExecutionError(
                    f"request {self.role} ended before {field}: {self.outcome()}"
                )
        value = getattr(self, field)
        if not isinstance(value, int):
            raise CaseExecutionError(f"request {self.role} signalled {field} without a timestamp")
        if report_wait:
            self._progress(
                "request.wait_done",
                target=target,
                waited_ms=(time.perf_counter_ns() - wait_started_ns) / 1e6,
            )
        return value

    def wait_sent(self, timeout: float) -> int:
        return self._wait_for(self._sent, "sent_ns", timeout)

    def wait_body_sent(self, timeout: float) -> int:
        return self._wait_for(self._body_sent, "body_sent_ns", timeout)

    def wait_accepted(self, timeout: float) -> int:
        return self._wait_for(self._accepted, "accepted_ns", timeout)

    def wait_first_output(self, timeout: float) -> int:
        return self._wait_for(self._first_output, "first_output_ns", timeout)

    def wait_completed(self, timeout: float) -> int:
        return self._wait_for(self._completed, "completed_ns", timeout)

    def wait_done(self, timeout: float) -> None:
        wait_started_ns = time.perf_counter_ns()
        if not self._done.is_set():
            self._progress("request.wait", target="worker_done", timeout_s=timeout)
        if not self._done.wait(timeout):
            self._progress(
                "request.wait_failed",
                target="worker_done",
                reason="timeout",
                waited_ms=(time.perf_counter_ns() - wait_started_ns) / 1e6,
            )
            raise CaseExecutionError(f"request {self.role} did not terminate")
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0)
        self._progress(
            "request.wait_done",
            target="worker_done",
            waited_ms=(time.perf_counter_ns() - wait_started_ns) / 1e6,
        )

    def outcome(self) -> str:
        with self._lock:
            result = self._result
            thread_error = self._thread_error
            first = self.first_output_ns
            terminal = self.completed_ns
        if thread_error is not None:
            return "worker_error"
        if result is None:
            if self.cancel_ns is not None:
                return "cancelled"
            return "running" if not self._done.is_set() else "worker_error"
        http = result.http
        if http.cancelled:
            return "cancelled"
        if http.error is not None or http.status is None:
            return "transport_error"
        if not (200 <= http.status < 300):
            return "rejected"
        if result.protocol_error is not None:
            return "protocol_error"
        if not self.stream:
            return "nonstream_success"
        if result.error_code is not None:
            return "failed_before_first" if first is None else "stream_error"
        if first is None:
            return "failed_before_first"
        terminal_events = [event for event in result.events if event.kind == "terminal"]
        if not terminal_events:
            return "incomplete_stream"
        return "success"

    def as_record(self) -> dict[str, Any]:
        with self._lock:
            result = self._result
            event_trace = list(self._event_trace)
            thread_error = self._thread_error
            output_bytes = len("".join(self._output_parts).encode("utf-8"))
        sent = self.sent_ns
        first = self.first_output_ns
        ttft_ns = first - sent if isinstance(sent, int) and isinstance(first, int) else None
        http = result.http if result is not None else None
        return {
            "role": self.role,
            "order": self.order,
            "protocol": self.protocol,
            "stream": self.stream,
            "body_bytes": self._prepared.body_bytes,
            "sent_ns": sent,
            "body_sent_ns": self.body_sent_ns,
            "accepted_ns": self.accepted_ns,
            "first_output_ns": first,
            "completed_ns": self.completed_ns,
            "cancel_ns": self.cancel_ns,
            "cancel_requested": (
                http.cancel_requested if http is not None else self.cancel_ns is not None
            ),
            "transport_cancelled": http.cancelled if http is not None else False,
            "ttft_ns": ttft_ns,
            "outcome": self.outcome(),
            "http_status": http.status if http is not None else None,
            "http_reason": http.reason if http is not None else None,
            "response_id": self.response_id,
            "output_bytes": output_bytes,
            "error_code": result.error_code if result is not None else None,
            "error_message": result.error_message if result is not None else None,
            "transport_error": http.error if http is not None else None,
            "protocol_error": result.protocol_error if result is not None else thread_error,
            "events": event_trace,
        }


class CaseContext:
    def __init__(
        self,
        client: NInferServeClient,
        model: str,
        timeout_seconds: float,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout_seconds
        self._on_progress = on_progress
        self.handles: list[RequestHandle] = []
        self.failures: list[FailedCondition] = []
        self.notes: dict[str, Any] = {}

    def progress(self, stage: str, **fields: Any) -> None:
        if self._on_progress is not None:
            self._on_progress(stage, time.perf_counter_ns(), fields)

    def prepare(self, role: str, request: ProtocolRequest) -> RequestHandle:
        order = len(self.handles)
        self.progress(
            "request.prepare",
            role=role,
            order=order,
            protocol=request.protocol,
            path=request.path,
            stream=request.stream,
        )
        try:
            prepared = self.client.prepare(request)
        except Exception as error:
            self.progress(
                "request.prepare_failed",
                role=role,
                order=order,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        handle = RequestHandle(role, order, prepared, self._on_progress)
        self.handles.append(handle)
        handle._progress(
            "request.prepared",
            protocol=request.protocol,
            path=request.path,
            stream=request.stream,
            body_bytes=prepared.body_bytes,
        )
        return handle

    def start(self, role: str, request: ProtocolRequest) -> RequestHandle:
        handle = self.prepare(role, request)
        handle.start()
        return handle

    def barrier(self, requests: Iterable[tuple[str, ProtocolRequest]]) -> list[RequestHandle]:
        request_list = list(requests)
        self.progress(
            "barrier.prepare",
            roles=[role for role, _ in request_list],
            count=len(request_list),
        )
        handles = [self.prepare(role, request) for role, request in request_list]
        gate = threading.Event()
        for handle in handles:
            handle.start(gate)
        self.progress("barrier.release", roles=[handle.role for handle in handles])
        gate.set()
        sent = [handle.wait_sent(self.timeout) for handle in handles]
        spread = max(sent) - min(sent) if sent else 0
        self.notes.setdefault("barrier_spread_ns", []).append(spread)
        self.progress(
            "barrier.sent",
            roles=[handle.role for handle in handles],
            spread_ms=spread / 1e6,
        )
        self.require(
            spread <= 5_000_000,
            "max(barrier.sent_ns)-min(barrier.sent_ns) <= 5ms",
            f"observed spread {spread / 1e6:.3f} ms",
        )
        return handles

    def require(self, condition: bool, expression: str, detail: str) -> None:
        if not condition:
            self.failures.append(FailedCondition(expression, detail))
            self.progress("case.condition_failed", expression=expression, detail=detail)

    def require_success(self, handle: RequestHandle) -> None:
        handle.wait_completed(self.timeout)
        handle.wait_done(self.timeout)
        outcome = handle.outcome()
        if outcome != "success":
            self.progress("case.prerequisite_failed", role=handle.role, outcome=outcome)
            raise CaseExecutionError(f"prerequisite {handle.role} failed: {outcome}")

    def wait_all(self, handles: Iterable[RequestHandle] | None = None) -> None:
        selected = list(handles) if handles is not None else list(self.handles)
        self.progress("case.wait_all", roles=[handle.role for handle in selected])
        for handle in selected:
            handle.wait_done(self.timeout)
        self.progress("case.wait_all_done", roles=[handle.role for handle in selected])

    def cancel_live(self) -> None:
        live = [handle.role for handle in self.handles if not handle.is_done]
        if live:
            self.progress("case.cleanup_cancel", roles=live)
        for handle in self.handles:
            if not handle.is_done:
                handle.cancel()
        for handle in self.handles:
            if not handle.is_done:
                try:
                    handle.wait_done(min(self.timeout, 5.0))
                except CaseExecutionError as error:
                    self.failures.append(
                        FailedCondition("cancelled worker terminated", str(error))
                    )
                    self.progress(
                        "case.cleanup_failed",
                        role=handle.role,
                        error=str(error),
                    )

    def records(self) -> list[dict[str, Any]]:
        return [handle.as_record() for handle in sorted(self.handles, key=lambda item: item.order)]
