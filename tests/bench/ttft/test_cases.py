from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

from tools.bench.ttft import cases


class Clock:
    now = 1_000_000_000

    def perf_counter_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Gate:
    def __init__(self, clock, trace):
        self.clock = clock
        self.trace = trace
        self.handle = None

    def set(self):
        handle = self.handle
        self.trace.append(("send", handle.role, self.clock.now))
        handle.sent_ns = self.clock.now + handle.lateness_ns


class Handle:
    def __init__(self, role, usage, lateness_ns, clock, trace):
        self.role = role
        self.usage = usage
        self.lateness_ns = lateness_ns
        self.sent_ns = None
        self.clock = clock
        self.trace = trace

    def start(self, gate=None):
        if gate is not None:
            gate.handle = self
        else:
            self.trace.append(("send", self.role, self.clock.now))
            self.sent_ns = self.clock.now + self.lateness_ns

    def as_record(self):
        return {"usage": self.usage}


class Context:
    model = "test-model"

    def __init__(self, clock, trace, usages):
        self.clock = clock
        self.trace = trace
        self.usages = usages
        self.notes = {}
        self.requests = []

    def prepare(self, role, request):
        self.trace.append(("prepare", role, self.clock.now))
        self.requests.append(request)
        # Deliberately finish request A's send after B's planned arrival. B must still be
        # released at its own deadline, without waiting for A to send or produce output.
        return Handle(
            role, self.usages[len(self.requests) - 1], 20_000_000 if role == "a" else 0,
            self.clock, self.trace,
        )

    def wait_all(self, handles):
        assert all(handle.sent_ns is not None for handle in handles)
        self.trace.append(("wait", None, self.clock.now))

    def require_success(self, handle, *, prerequisite):
        assert prerequisite is False
        self.trace.append(("success", handle.role, self.clock.now))


class Corpus:
    def shape_messages(self, name):
        assert name == "interferer-256"
        return [{"role": "system", "content": "Writer."},
                {"role": "user", "content": "Write a long story."}]


@pytest.mark.parametrize("mode", ["replay", "snapshot"])
def test_fixed_preemption_arrivals_do_not_wait_for_server_progress(monkeypatch, mode):
    clock = Clock()
    trace = []
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "threading", SimpleNamespace(Event=lambda: Gate(clock, trace)))
    context = Context(clock, trace, [
        {"input_tokens": 192, "output_tokens": 256},
        {"input_tokens": 192, "output_tokens": 256},
    ])
    definition = cases.get_case(f"preemption-{mode}")
    definition.run(context, Corpus())

    assert definition.profile == f"preemption-{mode}"
    assert [(event, role) for event, role, _ in trace[:4]] == [
        ("prepare", "a"), ("prepare", "b"), ("send", "a"), ("send", "b"),
    ]
    arrivals = context.notes["arrivals"]
    assert [entry["offset_ns"] for entry in arrivals] == [0, 10_000_000]
    assert arrivals[1]["released_ns"] - arrivals[0]["released_ns"] == 10_000_000
    assert arrivals[0]["sent_ns"] > arrivals[1]["sent_ns"]
    assert [entry["lateness_ns"] for entry in arrivals] == [20_000_000, 0]
    assert context.notes["arrival_mode"] == "fixed_schedule"
    assert context.notes["mechanism_requirements"] == ["preemption", f"{mode}_restore"]
    assert context.requests[0].payload["messages"] != context.requests[1].payload["messages"]
    for request in context.requests:
        assert request.payload["max_completion_tokens"] == 256
        assert "ignore_eos" not in request.payload
        assert "min_tokens" not in request.payload


def test_short_generation_and_missing_usage_remain_observations(monkeypatch):
    clock = Clock()
    trace = []
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "threading", SimpleNamespace(Event=lambda: Gate(clock, trace)))
    context = Context(clock, trace, [{"input_tokens": 188, "output_tokens": 3}, {}])
    cases.get_case("preemption-replay").run(context, Corpus())

    assert context.notes["observed_workload"] == [
        {"role": "a", "input_tokens": 188, "output_tokens": 3,
         "nominal_input_matched": False, "output_limit_reached": False},
        {"role": "b", "input_tokens": None, "output_tokens": None,
         "nominal_input_matched": None, "output_limit_reached": None},
    ]
    assert [role for event, role, _ in trace if event == "success"] == ["a", "b"]


@pytest.mark.parametrize("run", [cases._anonymous_hot, cases._session_hot])
def test_generated_continuations_are_not_labelled_as_matching_fixed_work(run):
    requests = []

    def start(role, request):
        requests.append(request)
        return SimpleNamespace(role=role, output_text="generated answer", response_id="resp_seed")

    context = SimpleNamespace(
        model="test-model", notes={"input_dependency": "fixed"}, start=start,
        require_success=lambda handle: None,
    )
    corpus = SimpleNamespace(
        shape_messages=lambda name: [{"role": "user", "content": "fixed seed input"}],
        shape=lambda name: {"max_output_tokens": 16},
    )
    run(context, corpus)

    assert len(requests) == 2
    assert context.notes["input_dependency"] == "generated"
    assert context.notes["throughput_comparable"] is False
    assert "generated assistant output" in context.notes["throughput_limitation"]
    continuation = requests[-1].payload
    if run is cases._anonymous_hot:
        assert {"role": "assistant", "content": "generated answer"} in continuation["messages"]
    else:
        assert continuation["previous_response_id"] == "resp_seed"


def test_live_log_waits_for_complete_lines_and_ignores_warmup(tmp_path):
    path = tmp_path / "request.jsonl"
    paused = {"event": "request_scheduling", "transition": "paused", "request": {"http_request_id": "req-1"}}
    resumed = {**paused, "transition": "snapshot_started"}
    path.write_text(json.dumps(paused) + "\n")
    reader = cases._LiveRequestLog(SimpleNamespace(notes={"request_log_jsonl": str(path)}))
    assert reader.poll() == []
    first = json.dumps(paused) + "\n"
    second = json.dumps(resumed) + "\n"
    with path.open("a") as output:
        output.write(first + second[:12])
    assert [item["event"] for item in reader.poll()] == [paused]
    with path.open("a") as output:
        output.write(second[12:])
    assert [item["event"] for item in reader.poll()] == [resumed]
    assert reader.poll() == []
    assert reader.samples == 2


class MechanismHandle:
    def __init__(self, role):
        self.role = role
        self.is_done = False
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        return 1000

    def outcome(self):
        return "cancelled" if self.cancelled else "success"

    def as_record(self):
        return {"wire_request_id": f"req-{self.role}"}


def mechanism_context():
    started = []
    def start(role, request):
        started.append((role, request))
        return MechanismHandle(role)
    def success(handle, *, prerequisite=True):
        assert handle.outcome() == "success"
    def require(condition, *args, **kwargs):
        assert condition, args
    return SimpleNamespace(model="test-model", notes={}, start=start,
                           require_success=success, require=require, started=started)


@pytest.mark.parametrize(("transitions", "expected_roles"), [
    ([], []),
    ([("paused", "replay"), ("restored", "snapshot")], []),
    ([("restored", "replay"), ("restored", "replay")], ["recovery-short-1", "recovery-short-2"]),
])
def test_recovery_arrivals_follow_the_first_replay_restore_only(
    monkeypatch, transitions, expected_roles,
):
    context = mechanism_context()
    checked = []
    context.require_success = lambda handle, **_kwargs: checked.append(handle.role)
    corpus = SimpleNamespace(shape_messages=lambda name: [
        {"role": "system", "content": "fixed reference"},
        {"role": "user", "content": "fixed question"},
    ])

    def fixed(ctx, requests, offsets, facts, on_sample):
        initial = [MechanismHandle(role) for role, _ in requests]
        for transition, route in transitions:
            on_sample({"event": {
                "event": "request_scheduling", "transition": transition, "route": route,
                "request": {"http_request_id": "req-b"},
            }, "observed_ns": 1000}, initial)
        return initial

    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("shared-growth-recovery").run(context, corpus)
    arrivals = [(role, request) for role, request in context.started if role != "seed"]
    assert [role for role, _ in arrivals] == expected_roles
    assert all(role in checked for role in expected_roles)
    assert all(request.payload["max_completion_tokens"] == 16 for _, request in arrivals)
    assert context.notes["throughput_comparable"] is False
    assert context.notes["arrival_mode"] == "fixed_initial_then_recovery_event"
    assert context.notes["mechanism_observations"]["replay_with_other_progress"] == "unavailable"
    if expected_roles:
        assert context.notes["mechanism_observations"]["recovery_arrival"] == "observed"
        assert context.notes["recovery_arrival_trigger"]["event"]["route"] == "replay"
    else:
        assert "recovery_arrival_trigger" not in context.notes


def test_pressure_cancellation_selects_the_identified_request_and_continues_probes(monkeypatch):
    context = mechanism_context()
    handles = [MechanismHandle(role) for role in ("a", "b", "c")]

    def fixed(ctx, requests, offsets, facts, on_sample):
        for wire in ("req-b", "req-c"):
            on_sample({"event": {
                "event": "request_scheduling", "transition": "paused", "request": {"http_request_id": wire},
            }, "observed_ns": 500}, handles)
        return handles

    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("host-history-pressure-cancel").run(context, Corpus())
    assert [handle.role for handle in handles if handle.cancelled] == ["b"]
    cancellation = context.notes["pressure_cancel"]
    assert cancellation["target_role"] == "b"
    assert cancellation["event"]["request"]["http_request_id"] == "req-b"
    assert context.notes["mechanism_observations"]["pressure_cancellation"] == "observed"
    assert [role for role, _ in context.started[-2:]] == ["history-0-probe", "history-1-probe"]


@pytest.mark.parametrize(("event", "already_done"), [
    ({"event": "throughput", "scheduler": {"paused": 1}}, False),
    ({"event": "request_scheduling", "transition": "paused", "request": {"http_request_id": "unrelated"}}, False),
    ({"event": "request_scheduling", "transition": "paused"}, False),
    ({"event": "request_scheduling", "transition": "snapshot_started", "request": {"http_request_id": "req-b"}}, False),
    ({"event": "request_scheduling", "transition": "paused", "request": {"http_request_id": "req-b"}}, True),
])
def test_pressure_cancellation_requires_a_live_target_pause(monkeypatch, event, already_done):
    context = mechanism_context()
    handles = [MechanismHandle(role) for role in ("a", "b", "c")]
    handles[0].as_record = lambda: {"wire_request_id": None}
    handles[1].is_done = already_done

    def fixed(ctx, requests, offsets, facts, on_sample):
        on_sample({"event": event, "observed_ns": 500}, handles)
        return handles

    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("host-history-pressure-cancel").run(context, Corpus())
    assert not any(handle.cancelled for handle in handles)
    assert "pressure_cancel" not in context.notes


@pytest.mark.parametrize("case_name", ["mixed-arrivals-burst", "mixed-arrivals-sparse"])
def test_mixed_arrivals_connect_at_their_deadlines_without_waiting_for_completions(
    monkeypatch, case_name,
):
    clock = Clock()
    trace = []
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "threading", SimpleNamespace(Event=lambda: Gate(clock, trace)))
    class ConnectingContext(Context):
        def prepare(self, role, request):
            handle = super().prepare(role, request)
            self.clock.sleep(0.003)
            return handle

    context = ConnectingContext(clock, trace, [{} for _ in range(12)])
    corpus = SimpleNamespace(
        shape_messages=lambda name: [
            {"role": "system", "content": "Writer."},
            {"role": "user", "content": name},
        ],
        state_messages=lambda label: [{"role": "user", "content": f"Reference {label}"}],
    )
    cases.get_case(case_name).run(context, corpus)
    arrivals = context.notes["arrivals"]
    sent = [(role, timestamp) for event, role, timestamp in trace if event == "send"]
    prepared = {role: timestamp for event, role, timestamp in trace if event == "prepare"}
    for arrival in arrivals:
        assert prepared[arrival["role"]] == arrival["planned_ns"]
        assert arrival["sent_ns"] - prepared[arrival["role"]] == 3_000_000
        assert arrival["lateness_ns"] == 3_000_000
    assert [timestamp - sent[0][1] for _, timestamp in sent] == [item["offset_ns"] for item in arrivals]
    assert all(event not in {"wait", "success"} for event, _, _ in trace[:2 * len(arrivals)])
    assert context.notes["workload_shape"]["eos_policy"] == "normal"
    assert {context.notes["request_classes"][role] for role, _ in sent} == {
        "growth", "long-prefill", "short", "medium-prefill",
    }
