"""Associate Serve request logs with wire requests and summarize runtime observations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SCHEDULING_FIELDS = (
    "preemptions", "snapshot_restores", "replay_restores", "replayed_tokens",
    "paused_ns", "device_to_host_bytes", "host_to_device_bytes",
)
MECHANISM_COUNTERS = {
    "preemption": "preemptions",
    "snapshot_restore": "snapshot_restores",
    "replay_restore": "replay_restores",
}

TRANSFER_RESOURCES = ("state", "main_kv", "backend_kv")
TRANSFER_DIRECTIONS = ("d2h", "h2d", "d2d")
OCCUPANCY_FIELDS = (
    "device_state_slots", "device_main_kv_pages", "device_backend_kv_pages",
    "host_state_slots", "host_kv_bytes", "host_context_occupied_bytes",
    "host_context_reserved_bytes",
)

TTFT_STAGES = ("prepare", "queue", "initial_binding", "paused", "resident", "http_residual")
ENGINE_HOST_PHASES = (
    "engine_boundary", "program_submit", "program_post", "engine_commit_output",
    "engine_maintenance", "total",
)
WORK_PHASES = ("submit", "wait", "post", "gpu")


def request_timing_analysis(request: dict[str, Any]) -> dict[str, Any]:
    """Separate the first-output wall partition from overlapping execution evidence."""
    diagnostics = request.get("diagnostics", {})
    result = diagnostics.get("result", {})
    ttft = request.get("ttft_ns")
    row: dict[str, Any] = {
        "analysis_status": "unavailable", "analysis_reason": None,
        "timing_boundary": "streaming_first_output" if request.get("stream") is True
                           else "aggregate_response",
        "ttft_ms": ttft / 1e6 if type(ttft) is int and ttft >= 0 else None,
        "prompt_tokens": result.get("prompt_tokens"),
        "cached_tokens": result.get("prefix_cache_hit_tokens"),
        "terminal_computed_prefill_tokens": result.get("computed_prefill_tokens"),
        "first_computed_prefill_tokens": None,
        "preferred_reused_tokens": None,
        "source_wait_ms": None,
        "revoked_checkpoints": None,
        "admission_fallback_reason": None,
        "engine_elapsed_ms": None,
        **{f"{stage}_{suffix}": None for stage in TTFT_STAGES for suffix in ("ms", "pct")},
        **{f"host_{phase}_ms": None for phase in ENGINE_HOST_PHASES},
        "device_wait_exposed_ms": None,
        **{f"{work}_{phase}_ms": None for work in ("prefill", "replay")
           for phase in (*WORK_PHASES, "program")},
        **{f"{resource}_{direction}_{field}": None for resource in TRANSFER_RESOURCES
           for direction in TRANSFER_DIRECTIONS for field in ("ms", "bytes")},
        **{f"prepare_{field}_ms": None for field in
           ("frontend", "acquisition", "media_preprocess", "media_preprocess_work", "tokenize")},
        **{f"{field}_before_first": None for field in
           ("preemptions", "snapshot_restores", "replay_restores", "replayed_tokens")},
    }

    def seconds(value: Any) -> float | None:
        return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    def read_seconds(source: Any, key: str) -> float | None:
        return seconds(source.get(key)) if isinstance(source, dict) else None

    def milliseconds(value: float | None) -> float | None:
        return value * 1000 if value is not None else None

    admission = diagnostics.get("admission")
    if isinstance(admission, dict):
        row["preferred_reused_tokens"] = admission.get("preferred_reused_tokens")
        row["source_wait_ms"] = milliseconds(read_seconds(admission, "source_wait_seconds"))
        row["revoked_checkpoints"] = admission.get("revoked_checkpoints")
        row["admission_fallback_reason"] = admission.get("fallback_reason")

    if request.get("stream") is not True:
        row["analysis_reason"] = "aggregate_response_is_terminal"
        return row
    if row["ttft_ms"] is None:
        row["analysis_reason"] = "no_client_first_output"
        return row
    if diagnostics.get("status") != "available":
        row["analysis_reason"] = diagnostics.get("reason", "request_diagnostics_unavailable")
        return row
    first = diagnostics.get("first_output_timing")
    if not isinstance(first, dict):
        row["analysis_reason"] = "first_output_timing_unavailable"
        return row

    engine = first.get("engine", {})
    scheduling = first.get("scheduling", {})
    prepare = read_seconds(diagnostics.get("terminal_timings_seconds"), "prepare")
    elapsed = read_seconds(first, "elapsed_seconds")
    queue = read_seconds(engine, "queue_wait_seconds")
    binding = read_seconds(first, "initial_binding_seconds")
    paused_ns = scheduling.get("paused_ns") if isinstance(scheduling, dict) else None
    paused = paused_ns / 1e9 if type(paused_ns) is int and paused_ns >= 0 else None
    row["engine_elapsed_ms"] = milliseconds(elapsed)
    values = {"prepare": prepare, "queue": queue, "initial_binding": binding, "paused": paused}
    if None not in (elapsed, queue, binding, paused):
        values["resident"] = elapsed - queue - binding - paused
    if None not in (prepare, elapsed):
        values["http_residual"] = ttft / 1e9 - prepare - elapsed
    for stage, value in values.items():
        row[f"{stage}_ms"] = milliseconds(value)
        row[f"{stage}_pct"] = value * 1e11 / ttft if value is not None and ttft > 0 else None

    missing = [stage for stage in TTFT_STAGES if row[f"{stage}_ms"] is None]
    negative = [stage for stage in TTFT_STAGES
                if row[f"{stage}_ms"] is not None and row[f"{stage}_ms"] < 0]
    if missing:
        row["analysis_reason"] = "missing_phase:" + ",".join(missing)
    elif negative:
        row["analysis_status"] = "inconsistent"
        row["analysis_reason"] = "negative_phase:" + ",".join(negative)
    else:
        row["analysis_status"] = "available"

    row["first_computed_prefill_tokens"] = first.get("computed_prefill_tokens")
    host = engine.get("host_exposed_seconds") if isinstance(engine, dict) else None
    for phase in ENGINE_HOST_PHASES:
        row[f"host_{phase}_ms"] = milliseconds(read_seconds(host, phase))
    row["device_wait_exposed_ms"] = milliseconds(read_seconds(engine, "device_wait_exposed_seconds"))
    for work in ("prefill", "replay"):
        timing = first.get(work)
        for phase in WORK_PHASES:
            row[f"{work}_{phase}_ms"] = milliseconds(read_seconds(timing, f"{phase}_seconds"))
        parts = [row[f"{work}_{phase}_ms"] for phase in ("submit", "wait", "post")]
        if all(part is not None for part in parts):
            row[f"{work}_program_ms"] = sum(parts)
    transfers = first.get("context_transfers", {})
    for resource in TRANSFER_RESOURCES:
        directions = transfers.get(resource, {}) if isinstance(transfers, dict) else {}
        for direction in TRANSFER_DIRECTIONS:
            transfer = directions.get(direction, {}) if isinstance(directions, dict) else {}
            row[f"{resource}_{direction}_ms"] = milliseconds(read_seconds(transfer, "seconds"))
            size = transfer.get("bytes") if isinstance(transfer, dict) else None
            row[f"{resource}_{direction}_bytes"] = size if type(size) is int and size >= 0 else None
    preparation = diagnostics.get("preparation_seconds")
    for field in ("frontend", "acquisition", "media_preprocess", "media_preprocess_work", "tokenize"):
        row[f"prepare_{field}_ms"] = milliseconds(read_seconds(
            preparation, "total" if field == "frontend" else field,
        ))
    for field in ("preemptions", "snapshot_restores", "replay_restores", "replayed_tokens"):
        value = scheduling.get(field) if isinstance(scheduling, dict) else None
        row[f"{field}_before_first"] = value if type(value) is int and value >= 0 else None
    return row


def _global_runtime_observations(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate server-wide interval deltas, independently of per-request attribution."""
    intervals = [event for event in events if event.get("event") == "throughput"]
    if not intervals:
        return {"status": "unavailable", "reason": "runtime_intervals_missing"}
    servers = {event.get("server_instance_id") for event in events}
    if len(servers) != 1 or not all(isinstance(server, str) and server for server in servers):
        return {"status": "unavailable", "reason": "ambiguous_server_instance"}
    if any(event.get("schema_version") not in (23, 24, 25) for event in intervals):
        return {"status": "unavailable", "reason": "unsupported_runtime_schema"}

    def values_at(path: Sequence[str]) -> list[int | float] | None:
        values = []
        for event in intervals:
            value: Any = event
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return None
            values.append(value)
        return values

    def total(*path: str) -> int | float | None:
        values = values_at(path)
        return sum(values) if values is not None else None

    transfers = {
        resource: {
            direction: {
                field: total("context_cache", f"{resource}_transfers", direction, field)
                for field in ("bytes", "seconds", "count" if resource == "state" else "pages")
            }
            for direction in TRANSFER_DIRECTIONS
        }
        for resource in TRANSFER_RESOURCES
    }
    host_work = {
        group: {field: total("host_work", group, field) for field in fields}
        for group, fields in (
            ("elapsed_seconds", ("engine_boundary", "program_submit", "program_post",
                                 "engine_commit_output", "engine_maintenance", "total")),
            ("work_class_seconds", ("decode_host", "decode_device_wait", "prefill_host",
                                    "prefill_device_wait", "control_host", "control_device_wait")),
            ("detail_subset_seconds", ("stats_publication",)),
            ("detail_invocations", ("stats_publication",)),
        )
    }
    host_work["device_wait_seconds"] = total("host_work", "device_wait_seconds")
    occupancy = {}
    for field in OCCUPANCY_FIELDS:
        values = values_at(("context_cache", "occupancy", field))
        occupancy[field] = {
            "sampled_max": max(values) if values is not None else None,
            "last": values[-1] if values is not None else None,
        }
    peak = values_at(("context_cache", "occupancy", "host_context_peak_occupied_bytes"))
    final_intervals = [index for index, event in enumerate(intervals)
                       if event.get("final_interval") is True]
    tail_status = (
        "observed" if final_intervals == [len(intervals) - 1]
        else "invalid" if final_intervals else "unconfirmed"
    )
    starts = [event for event in events if event.get("event") == "server_start"]
    startup = starts[0] if len(starts) == 1 else {}
    return {
        "status": "available", "server_instance_id": next(iter(servers)),
        "scope": "isolated_server_runtime_intervals", "intervals": len(intervals),
        "reported_interval_seconds": total("interval_seconds"),
        "shutdown_tail": tail_status,
        "transfers": transfers, "host_work": host_work, "occupancy": occupancy,
        "host_context_peak_occupied_bytes": max(peak) if peak is not None else None,
        "scheduling": {field: total("scheduling", field) for field in
                       ("preemptions", "snapshot_restores", "replay_restores", "replayed_tokens")},
        "startup_memory": startup.get("memory"),
        "startup_engine": startup.get("engine"),
        "environment": startup.get("environment"),
    }


def unavailable_diagnostics(reason: str, **identity: Any) -> dict[str, Any]:
    return {
        "status": "unavailable", "reason": reason,
        "mechanisms": {name: "unavailable" for name in MECHANISM_COUNTERS},
        **identity,
    }


def _scheduling_observations(
    requests: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare Engine-captured counters at Replay boundaries, independent of log delivery order."""
    fields = tuple(f"{scope}_{work}_tokens" for scope in ("global", "request")
                   for work in ("prefill", "decode", "replayed"))
    intervals = []
    incomplete_requests = []
    unavailable = False
    supported = any(event.get("schema_version") in (24, 25) for event in events)
    for request in requests:
        diagnostic = request.get("diagnostics", {})
        engine_id = diagnostic.get("engine_request_id")
        if diagnostic.get("status") != "available":
            unavailable = True
            incomplete_requests.append({"request_role": request.get("role"),
                                        "reason": "terminal_diagnostics_unavailable"})
            continue
        selected = [event for event in events
                    if event.get("event") == "request_scheduling"
                    and event.get("server_instance_id") == diagnostic.get("server_instance_id")
                    and event.get("engine_request_id") == engine_id
                    and event.get("request", {}).get("request_id") == diagnostic.get("service_request_id")]
        request["diagnostics"]["scheduling_transitions"] = sorted(
            selected, key=lambda event: event.get("steady_ns", -1))
        episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in selected:
            index = event.get("preemption_index")
            if type(index) is int and index > 0:
                episodes[index].append(event)
        restored_count = diagnostic.get("scheduling", {}).get("replay_restores", 0)
        complete_intervals = 0
        for index, transitions in sorted(episodes.items()):
            starts = [e for e in transitions if e.get("transition") == "restored" and e.get("route") == "replay"]
            ends = [e for e in transitions if e.get("transition") == "replay_complete"]
            if not starts and not ends:
                continue
            row = {"request_role": request.get("role"), "engine_request_id": engine_id,
                   "preemption_index": index, "status": "unavailable"}
            intervals.append(row)
            if len(starts) != 1 or len(ends) != 1:
                row["reason"] = "replay_interval_incomplete"
                unavailable = True
                continue
            start, end = starts[0], ends[0]
            begin_ns, end_ns = start.get("steady_ns"), end.get("steady_ns")
            before, after = start.get("progress", {}), end.get("progress", {})
            if (type(begin_ns) is not int or type(end_ns) is not int or end_ns < begin_ns
                    or any(type(before.get(field)) is not int or type(after.get(field)) is not int
                           or after[field] < before[field] for field in fields)):
                row["reason"] = "invalid_replay_interval"
                unavailable = True
                continue
            deltas = {field: after[field] - before[field] for field in fields}
            other = {work: deltas[f"global_{work}_tokens"] - deltas[f"request_{work}_tokens"]
                     for work in ("prefill", "decode", "replayed")}
            if any(value < 0 for value in other.values()):
                row["reason"] = "inconsistent_progress_counters"
                unavailable = True
                continue
            complete_intervals += 1
            row.update(status="available", duration_ns=end_ns - begin_ns, progress=deltas,
                       other_prefill_tokens=other["prefill"], other_decode_tokens=other["decode"],
                       other_replayed_tokens=other["replayed"],
                       other_new_progress=(deltas["request_replayed_tokens"] > 0
                                           and other["prefill"] + other["decode"] > 0))
        if complete_intervals != restored_count:
            unavailable = True
            incomplete_requests.append({"request_role": request.get("role"),
                                        "reason": "replay_interval_count_mismatch",
                                        "terminal_replay_restores": restored_count,
                                        "complete_intervals": complete_intervals})
    observed = any(row.get("other_new_progress") is True for row in intervals)
    return {
        "status": "available" if supported and not unavailable else "unavailable",
        "incomplete_requests": incomplete_requests,
        "replay_intervals": intervals,
        "mechanisms": {"replay_with_other_progress": "observed" if observed else
                       "unavailable" if not supported or unavailable else "not_observed"},
    }


def attach_generation_diagnostics(run: dict[str, Any], path: Path | None) -> str | None:
    """Join one isolated server log using wire identities, never submission order or time."""
    requests = run.get("requests", [])
    run["global_diagnostics"] = {"status": "unavailable", "reason": "request_log_missing"}
    if path is None or not path.is_file():
        for request in requests:
            request["diagnostics"] = unavailable_diagnostics("request_log_missing")
        return None
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_wire: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    runtime_events = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"line {line_number} is not an object")
                if event.get("artifact_type") != "ninfer_serve_request_log":
                    raise ValueError(f"line {line_number} has an unknown artifact type")
                server = event.get("server_instance_id")
                runtime_events.append(event)
                identity = event.get("request")
                if not isinstance(identity, dict) or not isinstance(server, str) or not server:
                    continue
                numeric_id = identity.get("request_id")
                if type(numeric_id) is not int:
                    continue
                key = (server, numeric_id)
                groups[key].append(event)
                for field in ("http_request_id", "response_id"):
                    value = identity.get(field)
                    if isinstance(value, str) and value:
                        by_wire[(field, value)].add(key)
    except (OSError, UnicodeError, ValueError) as error:
        run["global_diagnostics"] = {"status": "unavailable", "reason": "invalid_request_log"}
        for request in requests:
            request["diagnostics"] = unavailable_diagnostics("invalid_request_log")
        return f"cannot parse request log {path}: {error}"

    run["global_diagnostics"] = _global_runtime_observations(runtime_events)

    for request in requests:
        wire = {
            field: value for field, value in (
                ("http_request_id", request.get("wire_request_id")),
                ("response_id", request.get("response_id")),
            ) if isinstance(value, str) and value
        }
        candidates: set[tuple[str, int]] = set()
        for field, value in wire.items():
            candidates.update(by_wire.get((field, value), set()))
        if len(candidates) != 1:
            reason = "ambiguous_wire_identity" if candidates else "wire_identity_not_found"
            request["diagnostics"] = unavailable_diagnostics(reason)
            continue
        key = next(iter(candidates))
        events = groups[key]
        identities = {
            field: {
                event["request"][field] for event in events
                if isinstance(event["request"].get(field), str) and event["request"][field]
            }
            for field in ("http_request_id", "response_id")
        }
        if any(len(values) > 1 for values in identities.values()) or any(
            identities[field] and value not in identities[field] for field, value in wire.items()
        ):
            request["diagnostics"] = unavailable_diagnostics("conflicting_wire_identity")
            continue
        identity = {
            "server_instance_id": key[0], "service_request_id": key[1],
            "matched_by": [field for field, value in wire.items() if value in identities[field]],
        }
        starts = [item for item in events if item.get("event") == "request_start"
                  and item.get("schema_version") in (23, 24, 25)]
        preparation = starts[0].get("preparation_seconds") if len(starts) == 1 else None
        identity.update(
            preparation_seconds=preparation,
            preparation_status="available" if isinstance(preparation, dict) else "unavailable",
        )
        completed = [event for event in events if event.get("event") == "request_done"]
        if len(completed) != 1:
            reason = "duplicate_request_done" if completed else "request_done_missing"
            request["diagnostics"] = unavailable_diagnostics(reason, **identity)
            continue
        event = completed[0]
        generation = event.get("generation")
        scheduling = generation.get("scheduling") if isinstance(generation, dict) else None
        engine_id = generation.get("engine_request_id") if isinstance(generation, dict) else None
        if (
            event.get("schema_version") not in (23, 24, 25)
            or type(engine_id) is not int or engine_id <= 0
            or not isinstance(scheduling, dict)
            or any(type(scheduling.get(field)) is not int or scheduling[field] < 0
                   for field in SCHEDULING_FIELDS)
        ):
            request["diagnostics"] = unavailable_diagnostics("scheduling_unavailable", **identity)
            continue
        counts = {field: scheduling[field] for field in SCHEDULING_FIELDS}
        request["diagnostics"] = {
            "status": "available", **identity, "engine_request_id": engine_id,
            "scheduling": counts,
            "admission": generation.get("admission"),
            "mechanisms": {
                name: "observed" if counts[counter] > 0 else "not_observed"
                for name, counter in MECHANISM_COUNTERS.items()
            },
            "result": event.get("result", {}),
            "terminal_timings_seconds": event.get("timings_seconds"),
            "terminal_engine_timing": event.get("engine_timing"),
            "first_output_timing": event.get("first_output_timing"),
        }
    run["scheduling_observations"] = _scheduling_observations(requests, runtime_events)
    return None
