"""Render one TTFT campaign summary into its standard artifact bundle."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

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
    if not rows:
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
)


def _csv_row(kind: str, values: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, **{field: values.get(field) for field in CSV_FIELDS if field != "kind"}}


def render_csv(summary: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in summary["ttft_groups"]:
        writer.writerow(_csv_row("ttft", row))
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
    }
    json_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    csv_text = render_csv(summary)
    markdown_text = render_markdown(summary)
    try:
        _atomic_write(paths["json"], json_text)
        _atomic_write(paths["csv"], csv_text)
        _atomic_write(paths["markdown"], markdown_text)
    except OSError as error:
        raise ReportError(f"cannot write summary bundle under {campaign.root}: {error}") from error
    return summary, paths
