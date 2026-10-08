from __future__ import annotations

import json
import csv
import io
from pathlib import Path

import pytest

from tools.bench.ttft.cases import CaseDefinition, run_case
from tools.bench.ttft.execution import CaseContext, RequestHandle
from tools.bench.ttft.render import render_csv, render_markdown, render_request_analysis_csv
from tools.bench.ttft.report import CampaignData, PlannedRun, load_campaign, summarize_campaign
from tools.ninfer_serve.client import ProtocolEvent, ProtocolRequest, ServeExchangeResult
from tools.ninfer_serve.openai_chat import ChatStreamAdapter
from tools.streaming_http.client import HttpExchangeResult, HttpResponseHead
from tools.streaming_http.sse import SseDecoder


class PreparedResponse:
    body_bytes = 20
    body = b"{}"
    cancel_ns = None

    def __init__(self, events, *, protocol="openai_chat", error=None, cancelled=False):
        self.request = ProtocolRequest(protocol, "/test", {})
        self.events = events
        self.error = error
        self.cancelled = cancelled

    def execute(self, *, on_sent, on_body_sent, on_event, on_headers=None):
        on_sent(1_000_000)
        on_body_sent(2_000_000)
        if on_headers is not None:
            on_headers(HttpResponseHead(200, "OK", {"x-request-id": "req_wire"}, 2_500_000))
        for event in self.events:
            on_event(event)
        return ServeExchangeResult(
            protocol=self.request.protocol,
            body_bytes=self.body_bytes,
            http=HttpExchangeResult(
                sent_ns=1_000_000,
                body_sent_ns=2_000_000,
                ended_ns=max(20_000_000, max((event.received_ns for event in self.events),
                                           default=0) + 5_000_000),
                status=200,
                headers={"x-request-id": "req_wire"},
                error=self.error,
                cancelled=self.cancelled,
                cancel_requested=self.cancelled,
                cancel_ns=19_000_000 if self.cancelled else None,
            ),
            events=self.events,
        )


def record(events, **kwargs):
    handle = RequestHandle("subject", 0, PreparedResponse(events, **kwargs), None)
    handle.start()
    handle.wait_done(2)
    return handle.as_record()


def output(at, text="x"):
    return ProtocolEvent("model_output", "delta", at, output=text)


def test_stream_keeps_all_nonempty_events_and_distinguishes_terminal_from_eof():
    events = [
        ProtocolEvent("accepted", "assistant_role", 3_000_000, response_id="chatcmpl_a"),
        output(4_000_000, ""),
        output(5_000_000, "你"),
        output(5_000_000, "好"),
        output(12_000_000),
        ProtocolEvent("metadata", "chunk", 13_000_000,
                      payload={"usage": {"prompt_tokens": 42, "completion_tokens": 7}}),
        ProtocolEvent("terminal", "done", 15_000_000),
    ]
    result = record(events)
    assert result["outcome"] == "success"
    assert result["ttft_ns"] == 4_000_000
    assert result["output_event_count"] == 3
    assert result["output_gap_ns"] == [0, 7_000_000]
    assert result["output_bytes"] == 7
    assert result["terminal_ns"] == 15_000_000
    assert result["ended_ns"] == 20_000_000
    assert result["terminal_tail_ns"] == 3_000_000
    assert result["unterminated_tail_ns"] is None
    assert result["usage"] == {"input_tokens": 42, "output_tokens": 7}
    assert result["wire_request_id"] == "req_wire"
    assert result["response_id"] == "chatcmpl_a"


@pytest.mark.parametrize(
    ("ending", "kwargs", "outcome", "terminal"),
    [
        ([ProtocolEvent("error", "error", 15_000_000)], {}, "stream_error", 15_000_000),
        ([], {}, "incomplete_stream", None),
        ([], {"error": "timeout"}, "transport_error", None),
        ([], {"cancelled": True}, "cancelled", None),
    ],
)
def test_first_output_survives_each_later_failure(ending, kwargs, outcome, terminal):
    result = record([output(5_000_000), *ending], **kwargs)
    assert result["outcome"] == outcome
    assert result["ttft_ns"] == 4_000_000
    assert result["terminal_ns"] == terminal
    assert result["ended_ns"] == 20_000_000
    assert result["max_output_gap_ns"] is None
    assert result["unterminated_tail_ns"] == (15_000_000 if terminal is None else None)


def test_anthropic_usage_is_cumulative_and_partial_metadata_is_preserved():
    result = record([
        ProtocolEvent("accepted", "message_start", 3_000_000,
                      payload={"message": {"usage": {"input_tokens": 20, "output_tokens": 0}}}),
        output(5_000_000),
        ProtocolEvent("metadata", "message_delta", 10_000_000,
                      payload={"usage": {"output_tokens": 2}}),
        ProtocolEvent("metadata", "message_delta", 12_000_000,
                      payload={"usage": {"output_tokens": 5}, "delta": {"stop_reason": "max_tokens"}}),
        ProtocolEvent("terminal", "message_stop", 15_000_000),
    ], protocol="anthropic_messages")
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 5}
    assert result["events"][-2]["metadata"]["stop_reason"] == "max_tokens"


def test_fragmented_utf8_and_multiple_frames_keep_completion_timestamp():
    decoder = SseDecoder()
    adapter = ChatStreamAdapter()
    payload = {"id": "chatcmpl_a", "choices": [{"index": 0, "delta": {"content": "你"}}]}
    wire = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
    split = wire.index("你".encode()) + 1
    assert decoder.feed(wire[:split], 3_000_000) == []
    frames = decoder.feed(wire[split:] + wire, 8_000_000)
    events = [event for frame in frames for event in adapter.consume(frame)]
    result = record([*events, ProtocolEvent("terminal", "done", 15_000_000)])
    assert result["ttft_ns"] == 7_000_000
    assert result["output_gap_ns"] == [0]


def campaign(tmp_path: Path, records, *, notes=None, constructed=True):
    raw = tmp_path / "run.json"
    raw.write_text("{}")
    run = {
        "case": "test-case", "profile_label": "test-profile", "_sample": 1,
        "server": {"model": "test-model"}, "constructed": constructed,
        "status": "constructed" if constructed else "not_constructed",
        "requests": records, "notes": notes or {}, "_source": str(raw),
    }
    return CampaignData(
        root=tmp_path, manifest={"status": "complete"},
        plans=[PlannedRun(0, "test-case", "test-profile", 1, raw, raw, raw, None)],
        runs=[run], failures=[], artifact_errors=[],
    )


def test_report_shows_failed_and_unconstructed_ttft_without_counting_success(tmp_path):
    measured = record([output(5_000_000)], error="timeout")
    summary = summarize_campaign(campaign(tmp_path, [measured], constructed=False))
    assert summary["ttft_groups"] == []
    assert summary["stream_observations"][0]["ttft_ns"] == 4_000_000
    assert summary["stream_observations"][0]["outcome"] == "transport_error"
    assert summary["workload_metrics"][0]["throughput_comparable"] is False
    assert summary["workload_metrics"][0]["completed_output_tokens_per_second"] is None
    assert "transport_error" in render_markdown(summary)
    assert "request_observation" in render_csv(summary)


def test_workload_uses_final_usage_and_disables_dynamic_background_throughput(tmp_path):
    measured = record([
        output(5_000_000),
        ProtocolEvent("metadata", "chunk", 12_000_000,
                      payload={"usage": {"completion_tokens": 10}}),
        ProtocolEvent("terminal", "done", 15_000_000),
    ])
    summary = summarize_campaign(campaign(tmp_path, [measured]))
    assert summary["workload_metrics"][0]["completed_output_tokens_per_second"] == pytest.approx(10 / 0.019)
    disabled = summarize_campaign(campaign(tmp_path, [measured], notes={
        "throughput_comparable": False, "arrival_mode": "causal_dynamic_background",
    }))
    assert disabled["workload_metrics"][0]["completed_output_tokens_per_second"] is None
    measured["usage"] = {}
    missing = summarize_campaign(campaign(tmp_path, [measured]))
    assert missing["workload_metrics"][0]["completed_output_tokens_per_second"] is None
    assert missing["workload_metrics"][0]["missing_success_usage"] == 1


def test_request_failure_does_not_reclassify_completed_arrival_graph():
    context = CaseContext(None, "test-model", 2)
    def requests(ctx, corpus):
        handle = RequestHandle("subject", 0, PreparedResponse([output(5_000_000)]), None)
        ctx.handles.append(handle)
        handle.start()
        ctx.require_success(handle, prerequisite=False)
    definition = CaseDefinition("test", "openai_chat", "profile", "test", (), "test", requests)
    result = run_case(definition, context, object(), "profile")
    assert result["constructed"] is True
    assert result["request_checks_passed"] is False
    assert result["status"] == "request_failure"
    assert result["requests"][0]["outcome"] == "incomplete_stream"


def logged_campaign(tmp_path, records, events, *, required=()):
    raw = tmp_path / "run.json"
    log = tmp_path / "requests.jsonl"
    raw.write_text(json.dumps({
        "artifact_type": "ninfer_serve_ttft_run", "schema_version": 1,
        "case": "test-case", "profile_label": "test-profile", "constructed": True,
        "status": "constructed", "server": {"model": "test-model"}, "requests": records,
        "notes": {"mechanism_requirements": list(required)},
    }))
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "artifact_type": "ninfer_serve_ttft_campaign", "schema_version": 2,
        "run_count": 1, "case_count": 1,
        "plans": [{"case": "test-case", "profile": "test-profile", "sample": 1,
                   "raw": str(raw), "progress": str(tmp_path / "progress"),
                   "serve_log": str(tmp_path / "serve.log"), "request_log_jsonl": str(log)}],
    }))
    return load_campaign(tmp_path)


def done_event(service_id, wire_id, response_id, *, preemptions=0, replay=0):
    return {
        "artifact_type": "ninfer_serve_request_log", "schema_version": 23,
        "event": "request_done", "server_instance_id": "serve-test",
        "request": {"request_id": service_id, "http_request_id": wire_id,
                    "response_id": response_id},
        "result": {"completion_tokens": 8, "finish_reason": "output_limit"},
        "generation": {
            "engine_request_id": service_id + 10,
            "scheduling": {
                "preemptions": preemptions, "snapshot_restores": 0,
                "replay_restores": replay, "replayed_tokens": 192 if replay else 0,
                "paused_ns": 100_000_000 if preemptions else 0,
                "device_to_host_bytes": 0, "host_to_device_bytes": 0,
            },
        },
    }


def test_diagnostics_join_reversed_requests_by_wire_id_and_preserve_public_work(tmp_path):
    a = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    a.update(role="a", wire_request_id="wire-a", response_id="response-a",
             usage={"output_tokens": 8})
    b = {**a, "role": "b", "wire_request_id": "wire-b", "response_id": "response-b"}
    loaded = logged_campaign(tmp_path, [a, b], [
        done_event(2, "wire-b", "response-b", preemptions=1, replay=1),
        done_event(1, "wire-a", "response-a"),
    ], required=("preemption", "replay_restore"))
    summary = summarize_campaign(loaded)
    observed_a, observed_b = summary["stream_observations"]
    assert observed_a["service_request_id"] == 1
    assert observed_b["engine_request_id"] == 12
    assert observed_a["mechanisms"]["preemption"] == "not_observed"
    assert observed_b["mechanisms"]["replay_restore"] == "observed"
    assert observed_b["replayed_tokens"] == 192
    assert observed_b["usage_check"] == "matched"
    workload = summary["workload_metrics"][0]
    assert workload["required_mechanism_status"] == "observed"
    assert workload["measured_completed_output_tokens"] == 16
    assert workload["completed_output_tokens_per_second"] == pytest.approx(16 / 0.019)
    assert "Scheduling evidence" in render_markdown(summary)


@pytest.mark.parametrize("problem", ["legacy", "conflict", "duplicate", "missing_counters"])
def test_unavailable_diagnostics_never_become_zero_counters(tmp_path, problem):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    measured["response_id"] = "response-a"
    event = done_event(1, "req_wire", "response-a")
    if problem == "legacy":
        event["schema_version"] = 21
        event["request"] = {"request_id": 1}
    elif problem == "conflict":
        event["request"]["response_id"] = "different-response"
    elif problem == "missing_counters":
        del event["generation"]["scheduling"]["replayed_tokens"]
    events = [event, event] if problem == "duplicate" else [event]
    summary = summarize_campaign(logged_campaign(
        tmp_path, [measured], events, required=("preemption",),
    ))
    observed = summary["stream_observations"][0]
    assert observed["diagnostic_status"] == "unavailable"
    assert observed["preemptions"] is None
    assert observed["ttft_ns"] == 4_000_000
    assert summary["workload_metrics"][0]["required_mechanism_status"] == "unavailable"


def test_mechanism_not_observed_keeps_valid_performance(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    summary = summarize_campaign(logged_campaign(
        tmp_path, [measured], [done_event(1, "req_wire", "response-a")],
        required=("preemption", "snapshot_restore"),
    ))
    assert summary["workload_metrics"][0]["required_mechanism_status"] == "not_observed"
    assert summary["ttft_groups"][0]["median_ttft_ns"] == 4_000_000


def runtime_interval(*, schema=23, final=None, state_bytes=0, main_bytes=0,
                     host_seconds=0.0, occupied=0, peak=None):
    occupancy = {"device_state_slots": 1, "device_main_kv_pages": 8,
                 "device_backend_kv_pages": 0, "host_state_slots": 1,
                 "host_kv_bytes": occupied, "host_context_occupied_bytes": occupied,
                 "host_context_reserved_bytes": 0}
    if peak is not None:
        occupancy["host_context_peak_occupied_bytes"] = peak
    event = {
        "artifact_type": "ninfer_serve_request_log", "schema_version": schema,
        "event": "throughput", "server_instance_id": "serve-test",
        "interval_seconds": 0.125 if final else 1.0,
        "host_work": {"elapsed_seconds": {"total": host_seconds},
                      "device_wait_seconds": 0.25},
        "context_cache": {"occupancy": occupancy},
    }
    if final is not None:
        event["final_interval"] = final
    for resource, amount in (("state", state_bytes), ("main_kv", main_bytes), ("backend_kv", 0)):
        unit = "count" if resource == "state" else "pages"
        event["context_cache"][f"{resource}_transfers"] = {
            direction: {"bytes": amount if direction == "d2h" else 0,
                        "seconds": 0.001 if direction == "d2h" else 0.0, unit: 1}
            for direction in ("d2h", "h2d", "d2d")
        }
    return event


def test_global_intervals_include_unowned_demotion_and_shutdown_tail(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    summary = summarize_campaign(logged_campaign(tmp_path, [measured], [
        done_event(1, "req_wire", "response-a"),
        runtime_interval(final=False, main_bytes=100, host_seconds=0.1, occupied=800, peak=900),
        runtime_interval(final=True, state_bytes=25, main_bytes=50,
                         host_seconds=0.02, occupied=200, peak=1000),
    ]))
    global_work = summary["global_runtime_observations"][0]
    assert global_work["status"] == "available"
    assert global_work["shutdown_tail"] == "observed"
    assert global_work["intervals"] == 2
    assert global_work["reported_interval_seconds"] == 1.125
    assert global_work["transfers"]["main_kv"]["d2h"]["bytes"] == 150
    assert global_work["transfers"]["state"]["d2h"]["bytes"] == 25
    assert global_work["host_work"]["elapsed_seconds"]["total"] == pytest.approx(0.12)
    assert global_work["host_work"]["device_wait_seconds"] == 0.5
    assert global_work["occupancy"]["host_context_occupied_bytes"] == {"sampled_max": 800, "last": 200}
    assert global_work["host_context_peak_occupied_bytes"] == 1000
    assert summary["stream_observations"][0]["device_to_host_bytes"] == 0
    markdown = render_markdown(summary)
    assert "Global runtime costs" in markdown and "Sampled max" in markdown


def test_missing_runtime_fields_remain_explicit(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    event = runtime_interval(main_bytes=100)
    del event["context_cache"]["backend_kv_transfers"]["h2d"]["seconds"]
    summary = summarize_campaign(logged_campaign(tmp_path, [measured], [event]))
    global_work = summary["global_runtime_observations"][0]
    assert global_work["status"] == "available"
    assert global_work["shutdown_tail"] == "unconfirmed"
    assert global_work["host_context_peak_occupied_bytes"] is None
    assert global_work["transfers"]["main_kv"]["d2h"]["bytes"] == 100
    assert global_work["transfers"]["backend_kv"]["h2d"]["seconds"] is None
    assert summary["stream_observations"][0]["preemptions"] is None


def test_unsupported_runtime_schema_is_unavailable(tmp_path):
    summary = summarize_campaign(logged_campaign(tmp_path, [], [runtime_interval(schema=22)]))
    assert summary["global_runtime_observations"][0]["reason"] == "unsupported_runtime_schema"


def test_absent_or_mixed_server_runtime_data_is_unavailable(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    events = [done_event(1, "req_wire", "response-a")]
    data = logged_campaign(tmp_path, [measured], events)
    assert summarize_campaign(data)["global_runtime_observations"][0]["status"] == "unavailable"
    other = runtime_interval()
    other["server_instance_id"] = "another-server"
    data = logged_campaign(tmp_path, [measured], [*events, other])
    assert summarize_campaign(data)["global_runtime_observations"][0]["reason"] == "ambiguous_server_instance"


def test_generated_dependent_rate_is_observed_without_matched_input_claim(tmp_path):
    measured = record([
        output(5_000_000),
        ProtocolEvent("metadata", "chunk", 12_000_000,
                      payload={"usage": {"completion_tokens": 10}}),
        ProtocolEvent("terminal", "done", 15_000_000),
    ])
    summary = summarize_campaign(campaign(tmp_path, [measured], notes={
        "input_dependency": "generated", "throughput_comparable": False,
    }))
    workload = summary["workload_metrics"][0]
    assert workload["matched_input_eligible"] is False
    assert workload["observed_completed_output_tokens_per_second"] == pytest.approx(10 / 0.019)
    assert workload["completed_output_tokens_per_second"] is None


def test_matching_labels_do_not_claim_cross_campaign_conditions_verified(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    data = campaign(tmp_path, [measured])
    summary = summarize_campaign(data, data)
    assert summary["cross_campaign_comparisons"]
    assert summary["comparison_qualification"]["status"] == "conditions_unverified"
    assert "conditions_unverified" in render_markdown(summary)


def test_role_mechanism_cannot_borrow_another_requests_restore(tmp_path):
    vision = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    vision["role"] = "vision"
    vision["response_id"] = "response-vision"
    text = {**vision, "role": "text", "wire_request_id": "wire-text", "response_id": "response-text"}
    data = logged_campaign(tmp_path, [vision, text], [
        done_event(1, "req_wire", "response-vision"),
        done_event(2, "wire-text", "response-text", preemptions=1, replay=1),
    ])
    data.runs[0]["notes"].update({
        "mechanism_requirements": ["vision_replay", "replay_compute_overlap"],
        "mechanism_role_requirements": {"vision_replay": {"role": "vision", "counter": "replay_restores"}},
        "mechanism_observations": {"replay_compute_overlap": "unavailable"},
    })
    metrics = summarize_campaign(data)["workload_metrics"][0]
    assert metrics["mechanism_coverage"]["replay_restore"] == "observed"
    assert metrics["mechanism_coverage"]["vision_replay"] == "not_observed"
    assert metrics["mechanism_coverage"]["replay_compute_overlap"] == "unavailable"
    assert metrics["required_mechanism_status"] == "unavailable"


def first_output_event():
    event = done_event(1, "req_wire", "response-a", preemptions=9, replay=9)
    event["result"].update(prompt_tokens=100, prefix_cache_hit_tokens=80,
                           computed_prefill_tokens=20, generated_token_ids=[7, 8])
    event["timings_seconds"] = {"prepare": 0.020, "prefill": 9.0, "total": 10.0}
    event["engine_timing"] = {"queue_wait_seconds": 0.030}
    event["first_output_timing"] = {
        "elapsed_seconds": 0.070, "initial_binding_seconds": 0.010,
        "engine": {"queue_wait_seconds": 0.030,
                   "host_exposed_seconds": {"engine_boundary": 0.003, "total": 0.004},
                   "device_wait_exposed_seconds": 0.011},
        "scheduling": {"paused_ns": 5_000_000, "preemptions": 1, "snapshot_restores": 1,
                       "replay_restores": 0, "replayed_tokens": 0},
        "computed_prefill_tokens": 20,
        "prefill": {"submit_seconds": 0.002, "wait_seconds": 0.006,
                    "post_seconds": 0.001, "gpu_seconds": 0.007},
        "replay": {"submit_seconds": 0, "wait_seconds": 0,
                   "post_seconds": 0, "gpu_seconds": 0},
        "context_transfers": {"main_kv": {"h2d": {"bytes": 1024, "seconds": 0.008}}},
    }
    return event


def test_first_output_partition_preserves_terminal_and_overlapping_work(tmp_path):
    measured = record([output(101_000_000), ProtocolEvent("terminal", "done", 102_000_000)])
    done = first_output_event()
    done["schema_version"] = 25
    done["generation"]["admission"] = {
        "preferred_reused_tokens": 19057, "source_wait_seconds": 0.003,
        "revoked_checkpoints": 1, "fallback_reason": "source_revoked",
    }
    start = {**done, "event": "request_start",
             "preparation_seconds": {"total": 0.015, "acquisition": 0.003, "tokenize": 0.010}}
    loaded = logged_campaign(tmp_path, [measured], [start, done])
    diagnostics = loaded.runs[0]["requests"][0]["diagnostics"]
    assert diagnostics["result"]["generated_token_ids"] == [7, 8]
    assert diagnostics["terminal_timings_seconds"]["prefill"] == 9.0
    assert diagnostics["preparation_seconds"]["tokenize"] == 0.010
    summary = summarize_campaign(loaded)
    row = summary["request_timing_analysis"][0]
    assert row["preferred_reused_tokens"] == 19057
    assert row["source_wait_ms"] == 3.0
    assert row["revoked_checkpoints"] == 1
    assert row["admission_fallback_reason"] == "source_revoked"

    assert row["analysis_status"] == "available"
    assert row["ttft_ms"] == 100
    assert row["prepare_ms"] == 20
    assert row["queue_ms"] == 30
    assert row["initial_binding_ms"] == 10
    assert row["paused_ms"] == 5
    assert row["resident_ms"] == pytest.approx(25)
    assert row["http_residual_ms"] == pytest.approx(10)
    assert sum(row[f"{stage}_pct"] for stage in
               ("prepare", "queue", "initial_binding", "paused", "resident", "http_residual")) == pytest.approx(100)
    assert row["prefill_program_ms"] == 9
    assert row["prefill_gpu_ms"] == 7
    assert row["main_kv_h2d_ms"] == 8
    assert row["state_d2h_ms"] is None
    assert row["preemptions_before_first"] == 1
    assert summary["stream_observations"][0]["preemptions"] == 9
    csv_rows = list(csv.DictReader(io.StringIO(render_request_analysis_csv(summary))))
    assert csv_rows[0]["request_role"] == "subject"
    assert float(csv_rows[0]["resident_ms"]) == pytest.approx(25)
    assert "First-output latency sources" in render_markdown(summary)


@pytest.mark.parametrize("inconsistent", ["http_residual", "resident"])
def test_negative_wall_residual_is_retained(tmp_path, inconsistent):
    measured = record([output(101_000_000)])
    event = first_output_event()
    if inconsistent == "http_residual":
        event["timings_seconds"]["prepare"] = 0.031
    else:
        event["first_output_timing"]["initial_binding_seconds"] = 0.040
    row = summarize_campaign(logged_campaign(tmp_path, [measured], [event]))["request_timing_analysis"][0]
    assert row["analysis_status"] == "inconsistent"
    assert row[f"{inconsistent}_ms"] < 0
    assert row[f"{inconsistent}_pct"] < 0
    assert inconsistent in row["analysis_reason"]


@pytest.mark.parametrize("missing", ["output", "snapshot", "engine_phase", "aggregate", "logs"])
def test_unavailable_first_output_analysis_cannot_borrow_terminal_data(tmp_path, missing):
    measured = record([output(101_000_000)])
    event = first_output_event()
    if missing == "output":
        measured["ttft_ns"] = None
    elif missing == "snapshot":
        event["first_output_timing"] = None
    elif missing == "engine_phase":
        del event["first_output_timing"]["engine"]["queue_wait_seconds"]
    elif missing == "aggregate":
        measured["stream"] = False
    row = summarize_campaign(logged_campaign(
        tmp_path, [measured], [] if missing == "logs" else [event],
    ))["request_timing_analysis"][0]
    assert row["analysis_status"] == "unavailable"
    assert row["analysis_reason"]
    assert row["queue_ms"] is None


def test_raw_payload_media_is_deduplicated_and_reconstructable(tmp_path):
    from tools.bench.run_serve_ttft import _write

    data_url = "data:image/png;base64,aGVsbG8="
    prepared = PreparedResponse([output(5_000_000, "答案")])
    prepared.request.payload.update(messages=[{"content": [{"image_url": {"url": data_url}}]}])
    prepared.body = json.dumps(prepared.request.payload).encode()
    handle = RequestHandle("subject", 0, prepared, None)
    handle.start()
    handle.wait_done(2)
    raw = tmp_path / "sample-001.json"
    _write(raw, {"requests": [handle.as_record(freeze_input=True),
                             handle.as_record(freeze_input=True)]})
    records = json.loads(raw.read_text())["requests"]
    first = records[0]["request_payload"]["messages"][0]["content"][0]["image_url"]["url"]
    second = records[1]["request_payload"]["messages"][0]["content"][0]["image_url"]["url"]
    assert first == second
    assert first["type"] == "data_url_file" and first["encoding"] == "utf-8"
    assert (raw.parent / first["path"]).read_text() == data_url
    assert len(list((tmp_path / "sample-001-media").iterdir())) == 1
    assert records[0]["output_text"] == "答案"
    assert prepared.request.payload["messages"][0]["content"][0]["image_url"]["url"] == data_url


def test_final_record_retains_sent_body_when_input_object_changes():
    prepared = PreparedResponse([output(5_000_000)])
    prepared.body = b'{"messages":[{"content":"sent input"}]}'
    handle = RequestHandle("subject", 0, prepared, None)
    handle.start()
    handle.wait_done(2)
    prepared.request.payload.update(messages=[{"content": "later input"}])
    context = CaseContext(None, "test-model", 2)
    context.handles.append(handle)
    assert context.records()[0]["request_payload"]["messages"][0]["content"] == "sent input"


def test_lifecycle_class_and_phase_keep_slow_gaps_and_failed_requests(tmp_path):
    short = record([output(5_000_000), output(6_000_000),
                    ProtocolEvent("terminal", "done", 7_000_000)])
    short["role"] = "short"
    stalled = record([output(8_000_000), output(108_000_000)], error="timeout")
    stalled["role"] = "stalled"
    data = campaign(tmp_path, [short, stalled], notes={
        "request_classes": {"short": "decode", "stalled": "decode"},
        "workload_phases": {"seed": ["short"], "missing": ["absent"]},
    })
    summary = summarize_campaign(data)
    group = summary["request_lifecycle_groups"][0]
    assert group["requests"] == 2
    assert group["request_outcomes"] == {"success": 1, "transport_error": 1}
    assert group["output_gap"]["p95_ns"] == 100_000_000
    assert group["terminal"]["count"] == 1
    seed, missing = summary["workload_phase_metrics"]
    assert seed["output_gap"]["max_ns"] == 1_000_000
    assert seed["scope"] == "selected_client_requests"
    assert missing["missing_roles"] == ["absent"]
    assert missing["duration_ns"] is None
    assert "Gap p95/max ms" in render_markdown(summary)


def test_finite_schedule_drain_excludes_setup_and_uses_actual_usage(tmp_path):
    base = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    setup = {**base, "role": "setup", "sent_ns": 0, "ended_ns": 900,
             "usage": {"output_tokens": 1000}}
    first = {**base, "role": "first", "sent_ns": 100, "ended_ns": 500,
             "terminal_ns": 490, "usage": {"output_tokens": 3}}
    second = {**base, "role": "second", "sent_ns": 220, "ended_ns": 400,
              "terminal_ns": 380, "usage": {"output_tokens": 7}}
    summary = summarize_campaign(campaign(tmp_path, [setup, first, second], notes={
        "arrival_mode": "fixed_schedule",
        "arrivals": [{"role": "first", "offset_ns": 0, "lateness_ns": 0},
                     {"role": "second", "offset_ns": 100, "lateness_ns": 20}],
    }))
    workload = summary["workload_metrics"][0]
    assert workload["planned_injection_span_ns"] == 100
    assert workload["actual_injection_span_ns"] == 120
    assert workload["max_send_lateness_ns"] == 20
    assert workload["drain_ns"] == 280
    assert workload["terminal_drain_ns"] == 270
    assert workload["scheduled_completed_output_tokens"] == 10
    assert workload["scheduled_successful_requests"] == 2


def test_changed_contract_refuses_same_name_cross_campaign_delta(tmp_path):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    baseline = campaign(tmp_path, [measured])
    current = campaign(tmp_path, [measured], notes={"measurement_contract": "2"})
    summary = summarize_campaign(current, baseline)
    assert summary["cross_campaign_comparisons"] == []
    assert summary["cross_campaign_rejected"][0]["reason"] == "measurement_contract_mismatch"
    assert "Comparisons refused" in render_markdown(summary)


@pytest.mark.parametrize("missing_request", [False, True])
@pytest.mark.parametrize("other_decode", [0, 2])
def test_replay_progress_uses_engine_interval_and_subtracts_self(tmp_path, other_decode, missing_request):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    measured["response_id"] = "response-a"
    done = done_event(1, "req_wire", "response-a", preemptions=1, replay=1)
    done["schema_version"] = 24
    def transition(name, at, replayed, decoded):
        return {
            "artifact_type": "ninfer_serve_request_log", "schema_version": 24,
            "event": "request_scheduling", "server_instance_id": "serve-test",
            "request": {"request_id": 1, "http_request_id": "req_wire"},
            "engine_request_id": 11, "preemption_index": 1, "route": "replay",
            "transition": name, "steady_ns": at,
            "progress": {"global_prefill_tokens": 500, "global_decode_tokens": decoded,
                         "global_replayed_tokens": replayed, "request_prefill_tokens": 192,
                         "request_decode_tokens": 4, "request_replayed_tokens": replayed},
        }
    # Deliberately use timestamps unrelated to the client and reverse delivery order.
    end = transition("replay_complete", 200, 192, 40 + other_decode)
    start = transition("restored", 100, 0, 40)
    requests = [measured]
    if missing_request:
        requests.append({**measured, "role": "missing-terminal", "wire_request_id": "wire-missing",
                         "response_id": "response-missing"})
    expected = "observed" if other_decode else "unavailable" if missing_request else "not_observed"
    data = logged_campaign(tmp_path, requests, [end, done, start],
                           required=("replay_with_other_progress",))
    workload = summarize_campaign(data)["workload_metrics"][0]
    assert workload["required_mechanism_status"] == expected
    interval = workload["replay_progress_intervals"][0]
    assert interval["duration_ns"] == 100
    assert interval["other_decode_tokens"] == other_decode
    assert interval["other_replayed_tokens"] == 0


def test_engine_ticket_order_uses_exact_identity_not_client_send_order(tmp_path):
    a = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    a.update(role="older", wire_request_id="wire-a", response_id="response-a")
    b = {**a, "role": "younger", "wire_request_id": "wire-b", "response_id": "response-b"}
    data = logged_campaign(tmp_path, [a, b], [
        done_event(2, "wire-a", "response-a"), done_event(1, "wire-b", "response-b"),
    ], required=("expected_engine_order",))
    data.runs[0]["notes"]["expected_engine_order_groups"] = [["older"], ["younger"]]
    workload = summarize_campaign(data)["workload_metrics"][0]
    assert workload["required_mechanism_status"] == "not_observed"


@pytest.mark.parametrize("missing", ["restored", "replay_complete", "both", "terminal"])
def test_replay_evidence_missing_events_cannot_prove_no_other_progress(tmp_path, missing):
    measured = record([output(5_000_000), ProtocolEvent("terminal", "done", 15_000_000)])
    measured["response_id"] = "response-a"
    done = done_event(1, "req_wire", "response-a", preemptions=1, replay=1)
    done["schema_version"] = 24
    event = {
        "artifact_type": "ninfer_serve_request_log", "schema_version": 24,
        "event": "request_scheduling", "server_instance_id": "serve-test",
        "request": {"request_id": 1, "http_request_id": "req_wire"},
        "engine_request_id": 11, "preemption_index": 1, "route": "replay",
        "transition": "paused", "steady_ns": 50,
    }
    events = [event]
    if missing != "terminal":
        events.append(done)
    if missing not in ("restored", "both"):
        events.append({**event, "transition": "restored", "steady_ns": 100})
    if missing not in ("replay_complete", "both"):
        events.append({**event, "transition": "replay_complete", "steady_ns": 200})
    data = logged_campaign(tmp_path, [measured], events,
                           required=("replay_with_other_progress",))
    workload = summarize_campaign(data)["workload_metrics"][0]
    assert workload["required_mechanism_status"] == "unavailable"
    assert data.runs[0]["scheduling_observations"]["status"] == "unavailable"
    assert data.runs[0]["scheduling_observations"]["incomplete_requests"]
