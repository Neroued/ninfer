"""Render one TTFT campaign summary into its standard artifact bundle."""

from __future__ import annotations

import csv
import io
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.bench.ttft.diagnostics import TTFT_STAGES, request_timing_analysis
from tools.bench.ttft.report import ReportError, load_campaign, summarize_campaign


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _milliseconds(value: object, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    prefix = "+" if signed and value > 0 else ""
    absolute = abs(float(value))
    precision = 3 if absolute < 10 else 2 if absolute < 1000 else 1
    return f"{prefix}{float(value):.{precision}f}"


def _ratio(value: object) -> str:
    return f"{float(value):.3f}×" if isinstance(value, (int, float)) else "—"


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _link(label: str, path: object) -> str:
    if not isinstance(path, str) or not path:
        return "—"
    target = path.replace(" ", "%20").replace(")", "%29")
    return f"[{_cell(label)}]({target})"


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> list[str]:
    header = list(headers)
    result = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        result.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return result


def _comparison_markdown(summary: dict[str, Any]) -> list[str]:
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["comparisons"]:
        if row.get("available") is True:
            sections[str(row["section"])].append(row)
    lines = ["## Key comparisons", ""]
    if not sections:
        return [*lines, "No declared comparison has enough data yet.", ""]
    lines.extend(
        [
            "A positive delta means the subject is slower than its baseline. Ratios below 1× mean",
            "the subject is faster. Comparisons within one case are paired per run; cross-case",
            "comparisons use independent group medians.",
            "",
        ]
    )
    for section, rows in sections.items():
        lines.extend([f"### {section.title()}", ""])
        lines.extend(
            _table(
                ("Comparison", "Subject ms", "Baseline ms", "Delta ms", "Ratio", "N", "Method"),
                (
                    (
                        row["label"],
                        _milliseconds(row["subject_median_ttft_ms"]),
                        _milliseconds(row["baseline_median_ttft_ms"]),
                        _milliseconds(row["delta_ms"], signed=True),
                        _ratio(row["ratio"]),
                        row["compared_samples"],
                        "paired" if row["method"] == "paired_within_run" else "group medians",
                    )
                    for row in rows
                ),
            )
        )
        lines.append("")
    return lines


def _cross_campaign_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("cross_campaign_comparisons", [])
    rejected = summary.get("cross_campaign_rejected", [])
    if not rows and not rejected:
        return []
    baseline = summary["baseline"]
    ranked = sorted(
        rows,
        key=lambda row: abs(float(row["ratio"]) - 1.0) if row.get("ratio") is not None else -1.0,
        reverse=True,
    )
    lines = [
        "## Cross-campaign comparison",
        "",
        f"Baseline: `{baseline['root']}`",
        "",
        "The table shows the 30 largest relative changes among matching case observations.",
        "Fixed roles are matched by role; symmetric roles are matched by within-run TTFT rank.",
        "Qualification: conditions_unverified. Matching labels do not establish equal physical",
        "budgets, backend/Graph settings, observation overhead or generated-dependent inputs.",
        "",
    ]
    lines.extend(
        _table(
            (
                "Area",
                "Case",
                "Observation",
                "Current ms",
                "Baseline ms",
                "Delta ms",
                "Ratio",
            ),
            (
                (
                    row["category"],
                    row["case"],
                    row["observation_label"],
                    _milliseconds(row["current_median_ttft_ms"]),
                    _milliseconds(row["baseline_median_ttft_ms"]),
                    _milliseconds(row["delta_ms"], signed=True),
                    _ratio(row["ratio"]),
                )
                for row in ranked[:30]
            ),
        )
    )
    if rejected:
        lines.extend(["", "Comparisons refused because the measurement contract changed:", ""])
        lines.extend(_table(
            ("Case", "Profile", "Current contract", "Baseline contract"),
            ((row["case"], row["profile_label"], ", ".join(row["current_contracts"]),
              ", ".join(row["baseline_contracts"])) for row in rejected),
        ))
    lines.append("")
    return lines


def _ttft_markdown(summary: dict[str, Any]) -> list[str]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["ttft_groups"]:
        categories[str(row["category"])].append(row)
    lines = ["## TTFT by architecture area", ""]
    if not categories:
        return [*lines, "No successful constructed TTFT measurement is available.", ""]
    for category, rows in sorted(categories.items()):
        lines.extend([f"### {category.title()}", ""])
        lines.extend(
            _table(
                ("Case", "Role", "Median ms", "Min–max ms", "Samples", "Span", "Raw"),
                (
                    (
                        row["case"],
                        row["request_role"],
                        _milliseconds(row["median_ttft_ms"]),
                        f"{_milliseconds(row['min_ttft_ms'])}–{_milliseconds(row['max_ttft_ms'])}",
                        f"{row['samples']}/{row['expected_samples']}",
                        _percent(row["relative_span"]),
                        _link("open", row["raw_directory"]),
                    )
                    for row in rows
                ),
            )
        )
        lines.append("")
    return lines


def _symmetric_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary["symmetric_order_statistics"]
    if not rows:
        return []
    lines = [
        "## Symmetric arrival order statistics",
        "",
        "Symmetric roles are ranked inside each run; their arbitrary names are not treated as",
        "different workloads.",
        "",
    ]
    lines.extend(
        _table(
            ("Case", "Group", "Rank", "Median ms", "Min–max ms", "Role counts"),
            (
                (
                    row["case"],
                    row["symmetric_group"],
                    row["rank"],
                    _milliseconds(row["median_ttft_ms"]),
                    f"{_milliseconds(row['min_ttft_ms'])}–{_milliseconds(row['max_ttft_ms'])}",
                    json.dumps(
                        row["role_at_rank_counts"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
                for row in rows
            ),
        )
    )
    lines.append("")
    return lines


def _rejection_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary["boundary_rejections"]
    if not rows:
        return []
    lines = ["## Boundary rejections", ""]
    lines.extend(
        _table(
            ("Case", "Role", "HTTP", "Error code", "Samples"),
            (
                (
                    row["case"],
                    row["request_role"],
                    row["http_status"],
                    row["error_code"],
                    f"{row['samples']}/{row['expected_samples']}",
                )
                for row in rows
            ),
        )
    )
    lines.append("")
    return lines


def _variability_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary["variability"][:15]
    if not rows:
        return []
    lines = [
        "## Highest observed variability",
        "",
        "This is a ranking by `(max-min)/median`, not a pass/fail threshold. Fixed roles are",
        "tracked by role; symmetric roles are tracked by their within-run TTFT rank.",
        "",
    ]
    lines.extend(
        _table(
            ("Case", "Observation", "Median ms", "Min–max ms", "Relative span", "Raw"),
            (
                (
                    row["case"],
                    row["observation_label"],
                    _milliseconds(row["median_ttft_ms"]),
                    f"{_milliseconds(row['min_ttft_ms'])}–{_milliseconds(row['max_ttft_ms'])}",
                    _percent(row["relative_span"]),
                    _link("open", row["raw_directory"]),
                )
                for row in rows
            ),
        )
    )
    lines.append("")
    return lines


def _short_error(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    first = value.splitlines()[0]
    return first if len(first) <= 180 else first[:177] + "..."


def _diagnostic_links(row: dict[str, Any]) -> str:
    links = []
    for label, field in (
        ("raw", "raw"),
        ("progress", "progress"),
        ("serve", "serve_log"),
        ("request", "request_log_jsonl"),
    ):
        value = row.get(field)
        if isinstance(value, str):
            links.append(_link(label, value))
    return " · ".join(links) if links else "—"


def _issues_markdown(summary: dict[str, Any]) -> list[str]:
    failures = summary["failures"]
    not_constructed = summary["not_constructed_runs"]
    missing = summary["missing_runs"]
    invalid = summary["artifact_errors"]
    lines = ["## Structural issues", ""]
    if not any((failures, not_constructed, missing, invalid)):
        return [*lines, "No failed, missing, invalid, or unconstructed run was recorded.", ""]

    merged: dict[tuple[object, object], dict[str, Any]] = {}

    def entry(row: dict[str, Any]) -> dict[str, Any]:
        key = (row.get("case"), row.get("sample"))
        value = merged.setdefault(
            key,
            {
                "case": row.get("case", "—"),
                "sample": row.get("sample", "—"),
                "states": [],
                "details": [],
            },
        )
        for field in ("raw", "progress", "serve_log", "request_log_jsonl"):
            if isinstance(row.get(field), str):
                value[field] = row[field]
        return value

    for row in failures:
        value = entry(row)
        phase = row.get("phase")
        value["states"].append(f"failure:{phase}" if isinstance(phase, str) else "failure")
        if isinstance(row.get("error"), str):
            value["details"].append(row["error"])
    for row in not_constructed:
        value = entry(row)
        value["states"].append(str(row.get("status") or "not_constructed"))
        details = "; ".join(
            str(item.get("detail", ""))
            for item in row.get("failed_conditions", [])
            if isinstance(item, dict)
        )
        if details:
            value["details"].append(details)
    for row in missing:
        entry(row)["states"].append("missing_raw")
    for row in invalid:
        value = entry(row)
        value["states"].append("invalid_artifact")
        if isinstance(row.get("error"), str):
            value["details"].append(row["error"])

    lines.extend(
        _table(
            ("Case", "Sample", "State", "Detail", "Artifacts"),
            (
                (
                    row["case"],
                    row["sample"],
                    ", ".join(dict.fromkeys(row["states"])),
                    _short_error("; ".join(dict.fromkeys(row["details"]))),
                    _diagnostic_links(row),
                )
                for _, row in sorted(
                    merged.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
                )
            ),
        )
    )
    lines.append("")
    return lines


def _lifecycle_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "## Request latency and streaming progress", "",
        "Constructed runs are grouped by declared request class, or by role when no class is",
        "declared. Latencies pool requests across samples; output-gap percentiles pool nonempty",
        "output-event gaps. Outcomes include later failures and intentional cancellations.", "",
    ]
    def metric(row: dict[str, Any], name: str, stat: str) -> str:
        value = row[name][stat + "_ns"]
        return _milliseconds(value / 1e6 if value is not None else None)
    lines.extend(_table(
        ("Case", "Class/role", "Runs", "Outcomes", "TTFT p50/p95/max ms",
         "Terminal p50/p95/max ms", "Gap p95/max ms"),
        ((row["case"], f"{row['group_kind']}: {row['group_label']}", row["samples"],
          json.dumps(row["request_outcomes"], sort_keys=True),
          "/".join(metric(row, "ttft", stat) for stat in ("median", "p95", "max")),
          "/".join(metric(row, "terminal", stat) for stat in ("median", "p95", "max")),
          "/".join(metric(row, "output_gap", stat) for stat in ("p95", "max")))
         for row in summary["request_lifecycle_groups"]),
    ))
    phases = summary["workload_phase_metrics"]
    if phases:
        lines.extend(["", "### Selected workload phases", "",
                      "Only listed requests contribute to these client measurements. Server-wide work",
                      "and occupancy belong to the entire run and are not attributed to a phase.", ""])
        lines.extend(_table(
            ("Case", "Sample", "Phase", "Requests", "Missing roles", "Span ms",
             "TTFT mean/p50/p95/max ms", "Gap p95/max ms"),
            ((row["case"], row["sample"], row["phase"], row["requests"],
              ", ".join(row["missing_roles"]) or "—",
              _milliseconds(row["duration_ns"] / 1e6 if row["duration_ns"] is not None else None),
              "/".join(metric(row, "ttft", stat) for stat in ("mean", "median", "p95", "max")),
              "/".join(metric(row, "output_gap", stat) for stat in ("p95", "max")))
             for row in phases),
        ))
    lines.append("")
    return lines


def _stream_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "## Complete request observations", "",
        "TTFT remains observable after a later failure. Gaps are between nonempty output events,",
        "not tokens; multiple SSE frames read together can have equal timestamps. Terminal tail",
        "ends at the protocol terminal (including errors). An unterminated tail ends at transport",
        "closure without a protocol terminal. Bytes and events are not token counts.", "",
    ]
    def ms(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return _milliseconds(value / 1e6 if isinstance(value, (int, float)) else None)
    lines.extend(_table(
        ("Case", "Sample", "Role", "Constructed", "Outcome", "TTFT ms", "Terminal ms",
         "End ms", "P50 gap ms", "P95 gap ms", "Max gap ms", "Terminal tail ms",
         "Unterminated tail ms", "Events", "Input/output tokens"),
        (
            (row["case"], row["sample"], row["request_role"], row["constructed"], row["outcome"],
             ms(row, "ttft_ns"), ms(row, "terminal_latency_ns"), ms(row, "transport_duration_ns"),
             ms(row, "median_output_gap_ns"), ms(row, "p95_output_gap_ns"),
             ms(row, "max_output_gap_ns"), ms(row, "terminal_tail_ns"), ms(row, "unterminated_tail_ns"),
             row.get("output_event_count", "—"), f"{row.get('input_tokens')}/{row.get('output_tokens')}")
            for row in summary["stream_observations"]
        ),
    ))
    lines.extend([
        "", "## Workload completion", "",
        "Completed output throughput uses successful-request usage divided by the full workload",
        "makespan, including failed requests. Missing timing or successful-request usage leaves",
        "throughput unavailable. Observed rates remain visible for generated-dependent inputs",
        "and dynamic backgrounds; they are not matched-input throughput comparisons.", "",
    ])
    lines.extend(_table(
        ("Case", "Sample", "Arrival", "Input dependency", "Matched-input eligible", "Success/all", "Outcomes", "Makespan ms", "Completed tokens", "Observed tokens/s", "Limitation"),
        (
            (row["case"], row["sample"], row["arrival_mode"],
             row["input_dependency"], row["matched_input_eligible"],
             f"{row['successful_requests']}/{row['requests']}",
             json.dumps(row["request_outcomes"], sort_keys=True), ms(row, "duration_ns"),
             row["measured_completed_output_tokens"],
             f"{row['observed_completed_output_tokens_per_second']:.3f}" if row["observed_completed_output_tokens_per_second"] is not None else "—",
             row.get("throughput_limitation") or (f"Missing usage: {row['missing_success_usage']}" if row["missing_success_usage"] else "—"))
            for row in summary["workload_metrics"]
        ),
    ))
    scheduled = [row for row in summary["workload_metrics"] if "scheduled_requests" in row]
    if scheduled:
        lines.extend(["", "### Finite schedule and drain", "",
                      "Schedule metrics include only scheduled roles, excluding setup and later continuation.",
                      "Drain runs from the last actual send to the last transport end on the client clock.",
                      "A complete drain does not imply success; outcomes and completed token usage are separate.", ""])
        lines.extend(_table(
            ("Case", "Sample", "Success/scheduled", "Planned send span ms", "Actual send span ms",
             "Max send lateness ms", "Drain ms", "Completed output tokens"),
            ((row["case"], row["sample"],
              f"{row['scheduled_successful_requests']}/{row['scheduled_requests']}",
              ms(row, "planned_injection_span_ns"), ms(row, "actual_injection_span_ns"),
              ms(row, "max_send_lateness_ns"), ms(row, "drain_ns"),
              row["scheduled_completed_output_tokens"]) for row in scheduled),
        ))
    arrivals = []
    for workload in summary["workload_metrics"]:
        observed = {row["role"]: row for row in workload.get("observed_workload") or []}
        for arrival in workload.get("arrivals") or []:
            actual = observed.get(arrival["role"], {})
            arrivals.append((
                workload["case"], workload["sample"], arrival["role"],
                ms(arrival, "offset_ns"), ms(arrival, "lateness_ns"),
                actual.get("nominal_input_matched"), actual.get("output_limit_reached"),
            ))
    if arrivals:
        lines.extend(["", "### Fixed arrival schedule", ""])
        lines.extend(_table(
            ("Case", "Sample", "Role", "Planned offset ms", "Send lateness ms",
             "Nominal input matched", "Output limit reached"), arrivals,
        ))
    lines.extend([
        "", "## Scheduling evidence and costs", "",
        "Only exact wire request/response IDs join client observations to one server instance",
        "and service request. Missing or incomplete diagnostics are unavailable, not zero.",
        "A zero counter is not_observed; a positive counter is observed. Transfer bytes are",
        "per-generation totals and do not by themselves attribute a transfer to preemption.", "",
    ])
    lines.extend(_table(
        ("Case", "Sample", "Role", "Join", "Service/engine ID", "Preemption", "Snapshot", "Replay",
         "Replay tokens", "Paused ms", "D2H bytes", "H2D bytes", "Usage check", "Reason"),
        (
            (row["case"], row["sample"], row["request_role"], row["diagnostic_status"],
             f"{row['service_request_id']}/{row['engine_request_id']}",
             f"{row['mechanisms']['preemption']} ({row['preemptions']})",
             f"{row['mechanisms']['snapshot_restore']} ({row['snapshot_restores']})",
             f"{row['mechanisms']['replay_restore']} ({row['replay_restores']})",
             row["replayed_tokens"], ms(row, "paused_ns"), row["device_to_host_bytes"],
             row["host_to_device_bytes"], row["usage_check"], row["diagnostic_reason"] or "—")
            for row in summary["stream_observations"]
        ),
    ))
    lines.extend(["", "Required mechanisms do not change whether measured latency is retained.", ""])
    lines.extend(_table(
        ("Case", "Sample", "Requirements", "Coverage", "Individual evidence"),
        ((row["case"], row["sample"], ", ".join(row["mechanism_requirements"]) or "—",
          row["required_mechanism_status"],
          ", ".join(f"{name}={row['mechanism_coverage'].get(name, 'unavailable')}"
                    for name in row["mechanism_requirements"]) or "—")
         for row in summary["workload_metrics"]),
    ))
    intervals = [(workload, interval) for workload in summary["workload_metrics"]
                 for interval in workload.get("replay_progress_intervals", [])]
    if intervals:
        lines.extend(["", "### Progress during Replay", "",
                      "Counter differences are captured by Engine at restored and replay_complete.",
                      "Other new work subtracts this request's own work; log delivery order and",
                      "one-second scheduler samples do not establish this evidence.", ""])
        lines.extend(_table(
            ("Case", "Sample", "Role", "Episode", "Status", "Replay span ms",
             "Own replay tokens", "Other prefill/decode tokens", "Reason"),
            ((workload["case"], workload["sample"], row["request_role"], row["preemption_index"],
              row["status"], ms(row, "duration_ns"),
              row.get("progress", {}).get("request_replayed_tokens"),
              f"{row.get('other_prefill_tokens')}/{row.get('other_decode_tokens')}",
              row.get("reason", "—")) for workload, row in intervals),
        ))
    lines.append("")
    return lines


def _runtime_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary["global_runtime_observations"]
    lines = [
        "## Global runtime costs", "",
        "These are isolated-server interval deltas, including cache demotion without a request owner.",
        "The shutdown tail completes the interval stream when explicitly observed. Logs without",
        "a tail marker are unconfirmed; absent fields are unavailable, not zero.",
        "Host work is the mutually exclusive Engine clock, not a sum of request exposed times.",
        "Device wait and detail subsets must not be added to the Host total.", "",
    ]

    def seconds(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, (float, int)) else "—"

    lines.extend(_table(
        ("Case", "Sample", "Status", "Intervals", "Shutdown tail", "Reported seconds", "Host seconds", "Device wait seconds", "Reason"),
        ((row["case"], row["sample"], row["status"], row.get("intervals", "—"),
          row.get("shutdown_tail", "—"), seconds(row.get("reported_interval_seconds")),
          seconds(row.get("host_work", {}).get("elapsed_seconds", {}).get("total")),
          seconds(row.get("host_work", {}).get("scheduler_yield_seconds")), row.get("reason", "—"))
         for row in rows),
    ))
    available = [row for row in rows if row["status"] == "available"]
    if available:
        lines.extend(["", "### Physical transfers", ""])
        lines.extend(_table(
            ("Case", "Sample", "Resource", "Direction", "Bytes", "Seconds"),
            ((row["case"], row["sample"], resource, direction, values["bytes"], seconds(values["seconds"]))
             for row in available for resource, directions in {
                 "state": row["transfers"]["state"],
                 **{f"kv_pool_{pool}": values for pool, values in row["transfers"]["kv_pools"].items()},
             }.items()
             for direction, values in directions.items()),
        ))
        lines.extend([
            "", "### Resource occupancy", "",
            "Sampled maxima can miss short-lived allocations. Only the Host allocator's recorded",
            "lifetime high-water mark is labelled peak. Startup backing/capacity and the full Host",
            "work breakdown remain in the JSON report; sampled occupancy is not reserved backing.", "",
        ])
        lines.extend(_table(
            ("Case", "Sample", "Resource", "Sampled max", "Last sample"),
            ((row["case"], row["sample"], resource, value["sampled_max"], value["last"])
             for row in available for resource, value in {
                 **{key: value for key, value in row["occupancy"].items() if key != "kv_pools"},
                 **{f"kv_pool_{pool}": value for pool, value in row["occupancy"]["kv_pools"].items()},
             }.items()),
        ))
        lines.extend(["", *_table(
            ("Case", "Sample", "Host allocator peak occupied bytes"),
            ((row["case"], row["sample"], row["host_context_peak_occupied_bytes"]) for row in available),
        )])
    return [*lines, ""]


def _request_timing_markdown(summary: dict[str, Any]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["request_timing_analysis"]:
        groups[str(row["case"])].append(row)
    lines = [
        "## First-output latency sources", "",
        "Each row summarizes requests within one case; times are arithmetic means in milliseconds.",
        "The wall partition is prepare + queue + initial binding + paused + resident + HTTP residual.",
        "Negative residuals are retained and marked inconsistent. Missing logs or output, and",
        "aggregate responses, have no streaming first-output partition. Per-request milliseconds,",
        "TTFT percentages and execution details are in `request-analysis.csv` and the JSON report.",
        "Host exposure, prefill/replay work and CUDA transfer intervals overlap this partition;",
        "they are independent evidence and must not be added to it. Transfer intervals cover only",
        "completed request-owned work, excluding background cache reclamation.", "",
    ]

    def mean(rows: list[dict[str, Any]], field: str) -> str:
        values = [row[field] for row in rows if row.get(field) is not None]
        return f"{statistics.mean(values):.3f}" if values else "—"

    rows = []
    for case, requests in groups.items():
        measured = [row for row in requests if row["analysis_status"] != "unavailable"]
        inconsistent = sum(row["analysis_status"] == "inconsistent" for row in requests)
        rows.append((case, f"{len(measured)}/{len(requests)}", inconsistent,
                     mean(measured, "ttft_ms"),
                     *(mean(measured, f"{stage}_ms") for stage in TTFT_STAGES),
                     mean(measured, "host_total_ms"), mean(measured, "prefill_total_ms")))
    lines.extend(_table(
        ("Case", "Partitioned", "Inconsistent", "TTFT", "Prepare", "Queue", "Bind", "Paused",
         "Resident", "HTTP residual", "Host exposure", "Completed prefill interval"), rows,
    ))
    return [*lines, ""]


def render_markdown(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    coverage = summary["coverage"]
    lines = [
        "# NInfer Serve TTFT campaign summary",
        "",
        "This report summarizes externally observed HTTP time-to-first-token. Case names describe",
        "the constructed workload; the report does not infer private cache actions from latency.",
        "",
    ]
    lines.extend(
        _table(
            ("Campaign", "Status", "Model profile", "KV", "Runs", "Cases", "Failures"),
            (
                (
                    campaign.get("name"),
                    coverage.get("campaign_status"),
                    campaign.get("model_profile"),
                    campaign.get("kv_dtype"),
                    f"{coverage['constructed_runs']}/{coverage['planned_runs']} constructed",
                    f"{coverage['complete_cases']}/{coverage['planned_cases']} complete",
                    coverage["failure_records"],
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            f"Generated: `{summary['generated_at']}`",
            "",
            "## Coverage",
            "",
        ]
    )
    lines.extend(
        _table(
            (
                "Planned",
                "Present raw",
                "Valid raw",
                "Constructed",
                "Not constructed",
                "Missing",
                "Invalid",
            ),
            (
                (
                    coverage["planned_runs"],
                    coverage["present_artifacts"],
                    coverage["valid_artifacts"],
                    coverage["constructed_runs"],
                    coverage["not_constructed_runs"],
                    coverage["missing_artifacts"],
                    coverage["invalid_artifacts"],
                ),
            ),
        )
    )
    lines.append("")
    lines.extend(_comparison_markdown(summary))
    lines.extend(_cross_campaign_markdown(summary))
    lines.extend(_variability_markdown(summary))
    lines.extend(_rejection_markdown(summary))
    lines.extend(_issues_markdown(summary))
    lines.extend(_lifecycle_markdown(summary))
    lines.extend(_stream_markdown(summary))
    lines.extend(_request_timing_markdown(summary))
    lines.extend(_runtime_markdown(summary))
    lines.extend(_ttft_markdown(summary))
    lines.extend(_symmetric_markdown(summary))
    return "\n".join(lines).rstrip() + "\n"


CSV_FIELDS = (
    "kind",
    "section",
    "model",
    "category",
    "case",
    "profile_label",
    "request_role",
    "request_class",
    "observation_kind",
    "observation_label",
    "label",
    "subject_case",
    "subject_role",
    "baseline_case",
    "baseline_role",
    "method",
    "samples",
    "expected_samples",
    "subject_samples",
    "baseline_samples",
    "min_ttft_ns",
    "median_ttft_ns",
    "max_ttft_ns",
    "subject_median_ttft_ns",
    "baseline_median_ttft_ns",
    "delta_ns",
    "ratio",
    "speedup",
    "relative_span",
    "symmetric_group",
    "request_roles",
    "rank",
    "role_at_rank_counts",
    "http_status",
    "error_code",
    "raw_directory",
    "sample",
    "constructed",
    "outcome",
    "ttft_ns",
    "terminal_latency_ns",
    "transport_duration_ns",
    "max_output_gap_ns",
    "median_output_gap_ns",
    "p95_output_gap_ns",
    "terminal_tail_ns",
    "unterminated_tail_ns",
    "output_event_count",
    "output_bytes",
    "output_tokens",
    "wire_request_id",
    "response_id",
    "input_tokens",
    "finish_reason",
    "diagnostic_status",
    "diagnostic_reason",
    "server_instance_id",
    "service_request_id",
    "engine_request_id",
    "usage_check",
    "preemptions",
    "snapshot_restores",
    "replay_restores",
    "replayed_tokens",
    "paused_ns",
    "device_to_host_bytes",
    "host_to_device_bytes",
    "raw",
)


def _csv_row(kind: str, values: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, **{field: values.get(field) for field in CSV_FIELDS if field != "kind"}}


def render_csv(summary: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in summary["ttft_groups"]:
        writer.writerow(_csv_row("ttft", row))
    for row in summary["stream_observations"]:
        writer.writerow(_csv_row("request_observation", row))
    for row in summary["comparisons"]:
        if row.get("available") is not True:
            continue
        writer.writerow(
            _csv_row(
                "comparison",
                {
                    "section": row["section"],
                    "model": row["model"],
                    "label": row["label"],
                    "subject_case": row["subject"]["case"],
                    "subject_role": row["subject"]["role"],
                    "baseline_case": row["baseline"]["case"],
                    "baseline_role": row["baseline"]["role"],
                    "method": row["method"],
                    "samples": row["compared_samples"],
                    "subject_samples": row["subject_samples"],
                    "baseline_samples": row["baseline_samples"],
                    "subject_median_ttft_ns": row["subject_median_ttft_ns"],
                    "baseline_median_ttft_ns": row["baseline_median_ttft_ns"],
                    "delta_ns": row["delta_ns"],
                    "ratio": row["ratio"],
                    "speedup": row["speedup"],
                },
            )
        )
    for row in summary["symmetric_order_statistics"]:
        writer.writerow(
            _csv_row(
                "symmetric_order_stat",
                {
                    **row,
                    "request_roles": json.dumps(
                        row["request_roles"], separators=(",", ":")
                    ),
                    "role_at_rank_counts": json.dumps(
                        row["role_at_rank_counts"], separators=(",", ":")
                    ),
                },
            )
        )
    for row in summary["boundary_rejections"]:
        writer.writerow(_csv_row("rejection", row))
    for row in summary.get("cross_campaign_comparisons", []):
        writer.writerow(
            _csv_row(
                "cross_campaign",
                {
                    **row,
                    "samples": row["current_samples"],
                    "median_ttft_ns": row["current_median_ttft_ns"],
                    "baseline_median_ttft_ns": row["baseline_median_ttft_ns"],
                    "request_roles": (
                        json.dumps(row["request_roles"], separators=(",", ":"))
                        if row["request_roles"] is not None
                        else None
                    ),
                },
            )
        )
    return stream.getvalue()


def render_request_analysis_csv(summary: dict[str, Any]) -> str:
    fields = ["case", "profile_label", "sample", "request_role", "outcome", "constructed",
              "wire_request_id", "engine_request_id", *request_timing_analysis({})]
    fields.extend(sorted({key for row in summary["request_timing_analysis"] for key in row
                          if key not in fields and key != "raw"}))
    fields.append("raw")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary["request_timing_analysis"])
    return stream.getvalue()


def write_campaign_summary(
    campaign_dir: Path, baseline_dir: Path | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    campaign = load_campaign(campaign_dir)
    baseline = load_campaign(baseline_dir) if baseline_dir is not None else None
    summary = summarize_campaign(campaign, baseline)
    paths = {
        "json": campaign.root / "summary.json",
        "csv": campaign.root / "summary.csv",
        "markdown": campaign.root / "summary.md",
        "request_analysis": campaign.root / "request-analysis.csv",
    }
    json_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    csv_text = render_csv(summary)
    markdown_text = render_markdown(summary)
    request_csv_text = render_request_analysis_csv(summary)
    try:
        _atomic_write(paths["json"], json_text)
        _atomic_write(paths["csv"], csv_text)
        _atomic_write(paths["markdown"], markdown_text)
        _atomic_write(paths["request_analysis"], request_csv_text)
    except OSError as error:
        raise ReportError(f"cannot write summary bundle under {campaign.root}: {error}") from error
    return summary, paths
