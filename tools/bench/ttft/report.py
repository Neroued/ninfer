"""Pure aggregation for Serve TTFT run artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairDefinition:
    name: str
    left_case: str
    left_role: str
    right_case: str
    right_role: str


PAIRS = (
    PairDefinition("resume_state_host_minus_device", "resume-after-interference-state-host", "resume", "resume-after-interference-device", "resume"),
    PairDefinition("resume_kv_host_minus_device", "resume-after-interference-kv-host", "resume", "resume-after-interference-device", "resume"),
    PairDefinition("resume_both_host_minus_device", "resume-after-interference-both-host", "resume", "resume-after-interference-device", "resume"),
    PairDefinition("resume_evicted_minus_device", "resume-after-interference-evicted", "resume", "resume-after-interference-device", "resume"),
    PairDefinition("resume_catalog_minus_device", "resume-after-interference-catalog", "resume", "resume-after-interference-device", "resume"),
    PairDefinition("cache_off_minus_session_hot", "continuation-cache-off", "continuation", "session-hot-continuation", "continuation"),
    PairDefinition("unmarked_second_minus_first", "unmarked-common-prefix-miss", "second", "unmarked-common-prefix-miss", "first"),
    PairDefinition("shared_system_second_minus_first", "shared-sequential", "second", "shared-sequential", "first"),
    PairDefinition("shared_tools_second_minus_first", "shared-tools-sequential", "second", "shared-tools-sequential", "first"),
    PairDefinition("media_warm_second_minus_first", "media-preprocess-warm", "second", "media-preprocess-warm", "first"),
    PairDefinition("media_thrash_final_a_minus_warm_a", "media-cache-thrash", "a-final", "media-preprocess-warm", "second"),
    PairDefinition("prefill_128_short_minus_cold", "short-during-prefill-128", "short", "cold-short", "request"),
    PairDefinition("prefill_1024_short_minus_cold", "short-during-prefill-1024", "short", "cold-short", "request"),
    PairDefinition("prefill_4096_short_minus_cold", "short-during-prefill-4096", "short", "cold-short", "request"),
    PairDefinition("decode_short_minus_cold", "short-during-decode", "short", "cold-short", "request"),
    PairDefinition("media_prepare_short_minus_cold", "text-during-media-prepare", "short", "cold-short", "request"),
    PairDefinition("prefill_128_minus_1024", "short-during-prefill-128", "short", "short-during-prefill-1024", "short"),
    PairDefinition("prefill_4096_minus_1024", "short-during-prefill-4096", "short", "short-during-prefill-1024", "short"),
    PairDefinition("many_image_thread1_minus_default", "many-image-28-thread-1", "request", "many-image-28", "request"),
)


def load_runs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for supplied in paths:
        path = supplied.expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    runs: list[dict[str, Any]] = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReportError(f"cannot read {path}: {error}") from error
        if not isinstance(value, dict):
            raise ReportError(f"{path} does not contain a JSON object")
        if value.get("artifact_type") != "ninfer_serve_ttft_run" or value.get("schema_version") != 1:
            # Directories may also contain the summary produced by this module.
            if path.is_dir() or value.get("artifact_type") == "ninfer_serve_ttft_summary":
                continue
            raise ReportError(f"{path} is not a Serve TTFT run artifact")
        value["_source"] = str(path)
        runs.append(value)
    if not runs:
        raise ReportError("no Serve TTFT run artifacts found")
    return runs


def _stats(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "samples": len(ordered),
        "raw_ttft_ns": ordered,
        "min_ttft_ns": ordered[0],
        "median_ttft_ns": median,
        "max_ttft_ns": ordered[-1],
        "min_ttft_ms": ordered[0] / 1e6,
        "median_ttft_ms": median / 1e6,
        "max_ttft_ms": ordered[-1] / 1e6,
    }


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    symmetric_values: dict[
        tuple[str, str, str, str, tuple[str, ...], int], list[int]
    ] = defaultdict(list)
    symmetric_roles: dict[
        tuple[str, str, str, str, tuple[str, ...], int], Counter[str]
    ] = defaultdict(Counter)
    rejected: dict[tuple[str, str, str, str], Counter[tuple[Any, Any]]] = defaultdict(Counter)
    status_counts = Counter(run.get("status") for run in runs)
    for run in runs:
        case = run.get("case")
        profile = run.get("profile_label")
        server = run.get("server")
        model = server.get("model") if isinstance(server, dict) else None
        if not isinstance(case, str) or not isinstance(profile, str) or not isinstance(model, str):
            raise ReportError("run is missing case, profile_label, or public model identity")
        requests = run.get("requests", [])
        for request in requests:
            role = request.get("role")
            if not isinstance(role, str):
                raise ReportError(f"run {case} has a request without a role")
            if run.get("constructed") is True and request.get("outcome") == "success":
                ttft = request.get("ttft_ns")
                if not isinstance(ttft, int) or ttft < 0:
                    raise ReportError(f"successful request {case}/{role} has invalid TTFT")
                groups[(model, case, profile, role)].append(ttft)
            if request.get("outcome") == "rejected":
                rejected[(model, case, profile, role)][
                    (request.get("http_status"), request.get("error_code"))
                ] += 1

        if run.get("constructed") is True:
            request_by_role: dict[str, dict[str, Any]] = {}
            for request in requests:
                role = request.get("role")
                if isinstance(role, str):
                    if role in request_by_role:
                        raise ReportError(f"constructed run {case} repeats request role {role}")
                    request_by_role[role] = request
            for definition in run.get("symmetric_role_groups", []):
                if not isinstance(definition, dict):
                    raise ReportError(f"run {case} has an invalid symmetric role group")
                name = definition.get("name")
                raw_roles = definition.get("roles")
                if (
                    not isinstance(name, str)
                    or not isinstance(raw_roles, list)
                    or len(raw_roles) < 2
                    or not all(isinstance(role, str) for role in raw_roles)
                    or len(set(raw_roles)) != len(raw_roles)
                ):
                    raise ReportError(f"run {case} has an invalid symmetric role group")
                roles = tuple(raw_roles)
                ranked: list[tuple[int, str]] = []
                for role in roles:
                    request = request_by_role.get(role)
                    if request is None or request.get("outcome") != "success":
                        raise ReportError(
                            f"constructed run {case} lacks successful symmetric role {role}"
                        )
                    ttft = request.get("ttft_ns")
                    if not isinstance(ttft, int) or ttft < 0:
                        raise ReportError(
                            f"constructed run {case} has invalid symmetric TTFT for {role}"
                        )
                    ranked.append((ttft, role))
                ranked.sort()
                for rank, (ttft, role) in enumerate(ranked, start=1):
                    key = (model, case, profile, name, roles, rank)
                    symmetric_values[key].append(ttft)
                    symmetric_roles[key][role] += 1

    rows: list[dict[str, Any]] = []
    medians: dict[tuple[str, str, str], float] = {}
    for (model, case, profile, role), values in sorted(groups.items()):
        row = {
            "model": model,
            "case": case,
            "profile_label": profile,
            "request_role": role,
            **_stats(values),
        }
        rows.append(row)
        medians[(model, case, role)] = float(row["median_ttft_ns"])

    symmetric_rows: list[dict[str, Any]] = []
    for key, values in sorted(symmetric_values.items()):
        model, case, profile, name, roles, rank = key
        symmetric_rows.append(
            {
                "model": model,
                "case": case,
                "profile_label": profile,
                "symmetric_group": name,
                "request_roles": list(roles),
                "rank": rank,
                "role_at_rank_counts": dict(sorted(symmetric_roles[key].items())),
                **_stats(values),
            }
        )

    deltas = []
    models = sorted({model for model, _, _ in medians})
    for model in models:
        for pair in PAIRS:
            left = medians.get((model, pair.left_case, pair.left_role))
            right = medians.get((model, pair.right_case, pair.right_role))
            if left is None or right is None:
                continue
            delta = left - right
            deltas.append(
                {
                    "model": model,
                    "name": pair.name,
                    "left": {"case": pair.left_case, "role": pair.left_role, "median_ttft_ns": left},
                    "right": {"case": pair.right_case, "role": pair.right_role, "median_ttft_ns": right},
                    "median_delta_ns": delta,
                    "median_delta_ms": delta / 1e6,
                }
            )

    boundary_rows = []
    for (model, case, profile, role), counts in sorted(rejected.items()):
        for (status, code), count in sorted(counts.items(), key=lambda item: repr(item[0])):
            boundary_rows.append(
                {
                    "model": model,
                    "case": case,
                    "profile_label": profile,
                    "request_role": role,
                    "http_status": status,
                    "error_code": code,
                    "samples": count,
                }
            )

    return {
        "artifact_type": "ninfer_serve_ttft_summary",
        "schema_version": 1,
        "run_count": len(runs),
        "run_status_counts": dict(sorted(status_counts.items(), key=lambda item: str(item[0]))),
        "ttft_groups": rows,
        "symmetric_order_statistics": symmetric_rows,
        "paired_median_deltas": deltas,
        "boundary_rejections": boundary_rows,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "model",
        "case",
        "profile_label",
        "request_role",
        "symmetric_group",
        "request_roles",
        "rank",
        "role_at_rank_counts",
        "samples",
        "min_ttft_ns",
        "median_ttft_ns",
        "max_ttft_ns",
        "http_status",
        "error_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["ttft_groups"]:
            writer.writerow(
                {"kind": "ttft", **{field: row.get(field) for field in fields if field != "kind"}}
            )
        for row in summary["symmetric_order_statistics"]:
            serializable = {
                **row,
                "request_roles": json.dumps(row["request_roles"], separators=(",", ":")),
                "role_at_rank_counts": json.dumps(
                    row["role_at_rank_counts"], separators=(",", ":")
                ),
            }
            writer.writerow(
                {
                    "kind": "symmetric_order_stat",
                    **{field: serializable.get(field) for field in fields if field != "kind"},
                }
            )
        for row in summary["boundary_rejections"]:
            writer.writerow(
                {
                    "kind": "rejection",
                    **{field: row.get(field) for field in fields if field != "kind"},
                }
            )
