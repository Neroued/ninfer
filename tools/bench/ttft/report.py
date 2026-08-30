"""Campaign-aware aggregation for Serve TTFT run artifacts."""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.bench.ttft.cases import CASES


class ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetricRef:
    case: str
    role: str


@dataclass(frozen=True)
class ComparisonDefinition:
    name: str
    section: str
    label: str
    subject: MetricRef
    baseline: MetricRef


@dataclass(frozen=True)
class PlannedRun:
    index: int
    case: str
    profile: str
    sample: int
    raw: Path
    progress: Path
    serve_log: Path
    request_log_jsonl: Path | None


@dataclass
class CampaignData:
    root: Path
    manifest: dict[str, Any]
    plans: list[PlannedRun]
    runs: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    artifact_errors: list[dict[str, Any]]


def _ref(case: str, role: str) -> MetricRef:
    return MetricRef(case, role)


def _comparison(
    name: str,
    section: str,
    label: str,
    subject_case: str,
    subject_role: str,
    baseline_case: str,
    baseline_role: str,
) -> ComparisonDefinition:
    return ComparisonDefinition(
        name,
        section,
        label,
        _ref(subject_case, subject_role),
        _ref(baseline_case, baseline_role),
    )


# These are declared observations, not claims about private cache actions. A positive delta always
# means that the subject has higher external TTFT than its baseline.
COMPARISONS = (
    _comparison(
        "resume_state_host_vs_device",
        "resource pressure",
        "Host State resume vs Device",
        "resume-after-interference-state-host",
        "resume",
        "resume-after-interference-device",
        "resume",
    ),
    _comparison(
        "resume_kv_host_vs_device",
        "resource pressure",
        "Host KV resume vs Device",
        "resume-after-interference-kv-host",
        "resume",
        "resume-after-interference-device",
        "resume",
    ),
    _comparison(
        "resume_both_host_vs_device",
        "resource pressure",
        "Host State+KV resume vs Device",
        "resume-after-interference-both-host",
        "resume",
        "resume-after-interference-device",
        "resume",
    ),
    _comparison(
        "resume_evicted_vs_device",
        "resource pressure",
        "Evicted resume vs Device",
        "resume-after-interference-evicted",
        "resume",
        "resume-after-interference-device",
        "resume",
    ),
    _comparison(
        "resume_catalog_vs_device",
        "resource pressure",
        "Catalog pressure resume vs Device",
        "resume-after-interference-catalog",
        "resume",
        "resume-after-interference-device",
        "resume",
    ),
    _comparison(
        "session_hot_vs_cache_off",
        "private and session reuse",
        "Named hot continuation vs cache-off control",
        "session-hot-continuation",
        "continuation",
        "continuation-cache-off",
        "continuation",
    ),
    _comparison(
        "anonymous_continuation_vs_source",
        "private and session reuse",
        "Anonymous continuation vs its source",
        "anonymous-hot-continuation",
        "continuation",
        "anonymous-hot-continuation",
        "source",
    ),
    _comparison(
        "session_continuation_vs_source",
        "private and session reuse",
        "Named continuation vs its source",
        "session-hot-continuation",
        "continuation",
        "session-hot-continuation",
        "source",
    ),
    _comparison(
        "session_64k_a2_vs_a1",
        "resource pressure",
        "64K session A Host resume vs cold root",
        "session-alternating-64k-host-swap",
        "a2",
        "session-alternating-64k-host-swap",
        "a1",
    ),
    _comparison(
        "session_64k_b2_vs_b1",
        "resource pressure",
        "64K session B Host resume vs cold root",
        "session-alternating-64k-host-swap",
        "b2",
        "session-alternating-64k-host-swap",
        "b1",
    ),
    _comparison(
        "unmarked_second_vs_first",
        "private and session reuse",
        "Unmarked common-prefix second vs first",
        "unmarked-common-prefix-miss",
        "second",
        "unmarked-common-prefix-miss",
        "first",
    ),
    _comparison(
        "shared_system_second_vs_first",
        "shared prefix",
        "Marked Anthropic system-prefix second vs first",
        "shared-sequential",
        "second",
        "shared-sequential",
        "first",
    ),
    _comparison(
        "shared_openai_explicit_second_vs_first",
        "shared prefix",
        "OpenAI explicit system-prefix second vs first",
        "shared-openai-explicit",
        "second",
        "shared-openai-explicit",
        "first",
    ),
    _comparison(
        "shared_openai_implicit_reuse_vs_first",
        "shared prefix",
        "OpenAI implicit full-prompt reuse vs first",
        "shared-openai-implicit",
        "reuse",
        "shared-openai-implicit",
        "first",
    ),
    _comparison(
        "shared_observed_promotion_vs_first",
        "shared prefix",
        "Observed private-base promotion vs first",
        "shared-observed-promotion",
        "promotion",
        "shared-observed-promotion",
        "first-observation",
    ),
    _comparison(
        "shared_observed_reuse_vs_first",
        "shared prefix",
        "Observed shared reuse after private displacement vs first",
        "shared-observed-promotion",
        "shared-reuse",
        "shared-observed-promotion",
        "first-observation",
    ),
    _comparison(
        "shared_tools_second_vs_first",
        "shared prefix",
        "Stable tool-prefix second vs first",
        "shared-tools-sequential",
        "second",
        "shared-tools-sequential",
        "first",
    ),
    _comparison(
        "shared_tools_changed_vs_stable",
        "shared prefix",
        "Changed tool identity vs stable second use",
        "shared-tools-changed",
        "second",
        "shared-tools-sequential",
        "second",
    ),
    _comparison(
        "shared_slot_retention_final_vs_first",
        "shared prefix",
        "Incumbent prefix after equal-value competition vs first",
        "shared-slot-retention",
        "a-final",
        "shared-slot-retention",
        "a-first",
    ),
    _comparison(
        "shared_value_replacement_reuse_vs_first",
        "shared prefix",
        "Higher-value replacement reuse vs first",
        "shared-value-replacement",
        "b-reuse",
        "shared-value-replacement",
        "b-first",
    ),
    _comparison(
        "prefill_128_short_vs_cold",
        "scheduling",
        "Short arrival during 128-token prefill vs cold",
        "short-during-prefill-128",
        "short",
        "cold-short",
        "request",
    ),
    _comparison(
        "prefill_1024_short_vs_cold",
        "scheduling",
        "Short arrival during 1024-token prefill vs cold",
        "short-during-prefill-1024",
        "short",
        "cold-short",
        "request",
    ),
    _comparison(
        "prefill_4096_short_vs_cold",
        "scheduling",
        "Short arrival during 4096-token prefill vs cold",
        "short-during-prefill-4096",
        "short",
        "cold-short",
        "request",
    ),
    _comparison(
        "decode_short_vs_cold",
        "scheduling",
        "Short arrival during decode vs cold",
        "short-during-decode",
        "short",
        "cold-short",
        "request",
    ),
    _comparison(
        "media_prepare_short_vs_cold",
        "scheduling",
        "Short arrival during media preparation vs cold",
        "text-during-media-prepare",
        "short",
        "cold-short",
        "request",
    ),
    _comparison(
        "prefill_128_vs_1024",
        "scheduling",
        "128-token vs 1024-token prefill interference",
        "short-during-prefill-128",
        "short",
        "short-during-prefill-1024",
        "short",
    ),
    _comparison(
        "prefill_4096_vs_1024",
        "scheduling",
        "4096-token vs 1024-token prefill interference",
        "short-during-prefill-4096",
        "short",
        "short-during-prefill-1024",
        "short",
    ),
    _comparison(
        "media_prefix_continuation_vs_source",
        "media",
        "Exact media continuation vs source",
        "media-prefix-continuation",
        "continuation",
        "media-prefix-continuation",
        "source",
    ),
    _comparison(
        "media_prefix_append_vs_source",
        "media",
        "Media-prefix append vs source",
        "media-prefix-append",
        "continuation",
        "media-prefix-append",
        "source",
    ),
    _comparison(
        "media_prefix_changed_vs_source",
        "media",
        "Changed earlier media vs source",
        "media-prefix-changed",
        "changed-continuation",
        "media-prefix-changed",
        "source",
    ),
    _comparison(
        "media_warm_second_vs_first",
        "media",
        "Warm media preprocessing vs first use",
        "media-preprocess-warm",
        "second",
        "media-preprocess-warm",
        "first",
    ),
    _comparison(
        "media_thrash_final_vs_first",
        "media",
        "A after B/C pressure vs first A",
        "media-cache-thrash",
        "a-final",
        "media-cache-thrash",
        "a-first",
    ),
    _comparison(
        "many_image_thread1_vs_default",
        "media",
        "Single media worker vs default workers",
        "many-image-28-thread-1",
        "request",
        "many-image-28",
        "request",
    ),
    _comparison(
        "mixed_continuation_concurrent_vs_ordered",
        "mixed workload",
        "Concurrent vs ordered mixed continuation",
        "mixed-four-concurrent",
        "continuation",
        "mixed-four-ordered",
        "continuation",
    ),
    _comparison(
        "mixed_cold_long_concurrent_vs_ordered",
        "mixed workload",
        "Concurrent vs ordered mixed cold-long",
        "mixed-four-concurrent",
        "cold-long",
        "mixed-four-ordered",
        "cold-long",
    ),
    _comparison(
        "mixed_short_concurrent_vs_ordered",
        "mixed workload",
        "Concurrent vs ordered mixed short",
        "mixed-four-concurrent",
        "short",
        "mixed-four-ordered",
        "short",
    ),
    _comparison(
        "mixed_image_concurrent_vs_ordered",
        "mixed workload",
        "Concurrent vs ordered mixed image",
        "mixed-four-concurrent",
        "image",
        "mixed-four-ordered",
        "image",
    ),
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {path}: {error}") from error


def _relocate(root: Path, recorded_root: Path | None, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return root / path
    if recorded_root is not None:
        try:
            return root / path.relative_to(recorded_root)
        except ValueError:
            pass
    return path


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path)


def _required(value: dict[str, Any], name: str, expected: type, context: str) -> Any:
    result = value.get(name)
    if not isinstance(result, expected):
        raise ReportError(f"{context} has invalid {name}")
    return result


def _validate_run_artifact(value: dict[str, Any], plan: PlannedRun) -> None:
    context = f"run {plan.case} sample {plan.sample}"
    constructed = value.get("constructed")
    if not isinstance(constructed, bool):
        raise ReportError(f"{context} has no boolean constructed state")

    requests = value.get("requests")
    if not isinstance(requests, list):
        raise ReportError(f"{context} has no requests array")
    by_role: dict[str, dict[str, Any]] = {}
    for request in requests:
        if not isinstance(request, dict):
            raise ReportError(f"{context} has an invalid request")
        role = request.get("role")
        outcome = request.get("outcome")
        if not isinstance(role, str) or not role:
            raise ReportError(f"{context} has a request without a role")
        if role in by_role:
            raise ReportError(f"{context} repeats request role {role}")
        if not isinstance(outcome, str) or not outcome:
            raise ReportError(f"{context} request {role} has no outcome")
        if constructed and outcome == "success":
            ttft = request.get("ttft_ns")
            if not isinstance(ttft, int) or ttft < 0:
                raise ReportError(f"{context} request {role} has invalid TTFT")
        by_role[role] = request

    if constructed:
        server = value.get("server")
        model = server.get("model") if isinstance(server, dict) else None
        if not isinstance(model, str) or not model:
            raise ReportError(f"{context} has no public model identity")

    symmetric_groups = value.get("symmetric_role_groups", [])
    if not isinstance(symmetric_groups, list):
        raise ReportError(f"{context} has an invalid symmetric role group array")
    if not constructed:
        return
    for definition in symmetric_groups:
        if not isinstance(definition, dict):
            raise ReportError(f"{context} has an invalid symmetric role group")
        name = definition.get("name")
        roles = definition.get("roles")
        if (
            not isinstance(name, str)
            or not isinstance(roles, list)
            or len(roles) < 2
            or not all(isinstance(role, str) for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ReportError(f"{context} has an invalid symmetric role group")
        for role in roles:
            request = by_role.get(role)
            if request is None or request.get("outcome") != "success":
                raise ReportError(
                    f"{context} lacks successful symmetric role {role} in group {name}"
                )


def load_campaign(path: Path) -> CampaignData:
    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ReportError(f"{manifest_path} does not contain a JSON object")
    if (
        manifest.get("artifact_type") != "ninfer_serve_ttft_campaign"
        or manifest.get("schema_version") != 2
    ):
        raise ReportError(f"{manifest_path} is not a Serve TTFT campaign manifest")

    recorded_output = manifest.get("output_dir")
    recorded_root = (
        Path(recorded_output).expanduser()
        if isinstance(recorded_output, str)
        else None
    )
    raw_plans = manifest.get("plans")
    if not isinstance(raw_plans, list):
        raise ReportError(f"{manifest_path} has no plans array")

    plans: list[PlannedRun] = []
    keys: set[tuple[str, str, int]] = set()
    for index, value in enumerate(raw_plans, start=1):
        context = f"{manifest_path} plan {index}"
        if not isinstance(value, dict):
            raise ReportError(f"{context} is not an object")
        case = _required(value, "case", str, context)
        profile = _required(value, "profile", str, context)
        sample = _required(value, "sample", int, context)
        if sample <= 0:
            raise ReportError(f"{context} has non-positive sample")
        key = (case, profile, sample)
        if key in keys:
            raise ReportError(f"{manifest_path} repeats plan {key}")
        keys.add(key)
        raw = _relocate(root, recorded_root, _required(value, "raw", str, context))
        progress = _relocate(
            root, recorded_root, _required(value, "progress", str, context)
        )
        serve_log = _relocate(
            root, recorded_root, _required(value, "serve_log", str, context)
        )
        request_log_value = value.get("request_log_jsonl")
        if request_log_value is None:
            request_log_jsonl = None
        elif isinstance(request_log_value, str):
            request_log_jsonl = _relocate(root, recorded_root, request_log_value)
        else:
            raise ReportError(f"{context} has invalid request_log_jsonl")
        plans.append(
            PlannedRun(
                index,
                case,
                profile,
                sample,
                raw,
                progress,
                serve_log,
                request_log_jsonl,
            )
        )

    if manifest.get("run_count") != len(plans):
        raise ReportError(
            f"{manifest_path} run_count={manifest.get('run_count')!r}, "
            f"but plans contains {len(plans)} runs"
        )
    planned_case_count = len({plan.case for plan in plans})
    if manifest.get("case_count") != planned_case_count:
        raise ReportError(
            f"{manifest_path} case_count={manifest.get('case_count')!r}, "
            f"but plans contains {planned_case_count} cases"
        )

    runs: list[dict[str, Any]] = []
    artifact_errors: list[dict[str, Any]] = []
    for plan in plans:
        if not plan.raw.is_file():
            continue
        try:
            value = _read_json(plan.raw)
            if not isinstance(value, dict):
                raise ReportError("artifact is not a JSON object")
            if (
                value.get("artifact_type") != "ninfer_serve_ttft_run"
                or value.get("schema_version") != 1
            ):
                raise ReportError("artifact type/schema is not ninfer_serve_ttft_run v1")
            if value.get("case") != plan.case or value.get("profile_label") != plan.profile:
                raise ReportError("artifact case/profile does not match its campaign plan")
            _validate_run_artifact(value, plan)
        except ReportError as error:
            artifact_error = {
                "kind": "run",
                "case": plan.case,
                "profile_label": plan.profile,
                "sample": plan.sample,
                "raw": _relative(root, plan.raw),
                "progress": _relative(root, plan.progress),
                "serve_log": _relative(root, plan.serve_log),
                "error": str(error),
            }
            if plan.request_log_jsonl is not None:
                artifact_error["request_log_jsonl"] = _relative(
                    root, plan.request_log_jsonl
                )
            artifact_errors.append(artifact_error)
            continue
        value["_sample"] = plan.sample
        value["_source"] = _relative(root, plan.raw)
        value["_progress"] = _relative(root, plan.progress)
        value["_serve_log"] = _relative(root, plan.serve_log)
        value["_request_log_jsonl"] = (
            _relative(root, plan.request_log_jsonl)
            if plan.request_log_jsonl is not None
            else None
        )
        value["_plan_index"] = plan.index
        runs.append(value)

    failures: list[dict[str, Any]] = []
    failures_path = root / "failures.json"
    if failures_path.is_file():
        try:
            raw_failures = _read_json(failures_path)
            if not isinstance(raw_failures, list):
                raise ReportError("failures.json is not an array")
            for item in raw_failures:
                if not isinstance(item, dict):
                    raise ReportError("failures.json contains a non-object entry")
                normalized = dict(item)
                for field in ("raw", "progress", "serve_log", "request_log_jsonl"):
                    field_value = normalized.get(field)
                    if isinstance(field_value, str):
                        normalized[field] = _relative(
                            root, _relocate(root, recorded_root, field_value)
                        )
                failures.append(normalized)
        except ReportError as error:
            artifact_errors.append(
                {
                    "kind": "failures",
                    "case": None,
                    "profile_label": None,
                    "sample": None,
                    "raw": "failures.json",
                    "error": str(error),
                }
            )

    return CampaignData(root, manifest, plans, runs, failures, artifact_errors)


def _stats(samples: Sequence[tuple[int, int, str]]) -> dict[str, Any]:
    values = sorted(value for _, value, _ in samples)
    median = float(statistics.median(values))
    minimum = values[0]
    maximum = values[-1]
    mad = float(statistics.median(abs(value - median) for value in values))
    return {
        "samples": len(values),
        "sample_values": [
            {"sample": sample, "ttft_ns": value, "raw": source}
            for sample, value, source in sorted(samples)
        ],
        "raw_ttft_ns": values,
        "min_ttft_ns": minimum,
        "median_ttft_ns": median,
        "max_ttft_ns": maximum,
        "min_ttft_ms": minimum / 1e6,
        "median_ttft_ms": median / 1e6,
        "max_ttft_ms": maximum / 1e6,
        "median_absolute_deviation_ns": mad,
        "median_absolute_deviation_ms": mad / 1e6,
        "relative_span": (maximum - minimum) / median if median > 0 else None,
    }


def _case_metadata(case: str, runs: Sequence[dict[str, Any]]) -> tuple[str, str]:
    definition = CASES.get(case)
    if definition is not None:
        return definition.category, definition.description
    for run in runs:
        if run.get("case") == case:
            category = run.get("category")
            description = run.get("description")
            if isinstance(category, str) and isinstance(description, str):
                return category, description
    return "unknown", "Unknown case."


def _request_maps(
    runs: Sequence[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, str], list[tuple[int, int, str]]],
    dict[tuple[str, str, int], dict[str, int]],
    dict[tuple[str, str, str, str], Counter[tuple[Any, Any]]],
]:
    groups: dict[tuple[str, str, str, str], list[tuple[int, int, str]]] = defaultdict(list)
    run_roles: dict[tuple[str, str, int], dict[str, int]] = {}
    rejected: dict[tuple[str, str, str, str], Counter[tuple[Any, Any]]] = defaultdict(
        Counter
    )
    for run in runs:
        case = run.get("case")
        profile = run.get("profile_label")
        sample = run.get("_sample")
        server = run.get("server")
        model = server.get("model") if isinstance(server, dict) else None
        if not isinstance(case, str) or not isinstance(profile, str) or not isinstance(sample, int):
            raise ReportError("run is missing campaign case/profile/sample provenance")
        requests = run.get("requests")
        if not isinstance(requests, list):
            raise ReportError(f"run {case} sample {sample} has no requests array")
        by_role: dict[str, int] = {}
        seen_roles: set[str] = set()
        for request in requests:
            if not isinstance(request, dict):
                raise ReportError(f"run {case} sample {sample} has an invalid request")
            role = request.get("role")
            if not isinstance(role, str):
                raise ReportError(f"run {case} sample {sample} has a request without a role")
            if role in seen_roles:
                raise ReportError(f"run {case} sample {sample} repeats request role {role}")
            seen_roles.add(role)
            if request.get("outcome") == "success" and run.get("constructed") is True:
                ttft = request.get("ttft_ns")
                if not isinstance(ttft, int) or ttft < 0:
                    raise ReportError(
                        f"successful request {case}/{role} sample {sample} has invalid TTFT"
                    )
                if not isinstance(model, str) or not model:
                    raise ReportError(f"constructed run {case} has no public model identity")
                by_role[role] = ttft
                groups[(model, case, profile, role)].append(
                    (sample, ttft, str(run["_source"]))
                )
            if request.get("outcome") == "rejected" and run.get("constructed") is True:
                if not isinstance(model, str) or not model:
                    raise ReportError(f"rejected run {case} has no public model identity")
                rejected[(model, case, profile, role)][
                    (request.get("http_status"), request.get("error_code"))
                ] += 1
        if isinstance(model, str) and run.get("constructed") is True:
            run_roles[(model, case, sample)] = by_role
    return groups, run_roles, rejected


def _symmetric_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[
        tuple[str, str, str, str, tuple[str, ...], int], list[tuple[int, int, str]]
    ] = defaultdict(list)
    roles_at_rank: dict[
        tuple[str, str, str, str, tuple[str, ...], int], Counter[str]
    ] = defaultdict(Counter)
    for run in runs:
        if run.get("constructed") is not True:
            continue
        case = str(run["case"])
        profile = str(run["profile_label"])
        sample = int(run["_sample"])
        model = run["server"].get("model")
        if not isinstance(model, str):
            raise ReportError(f"constructed run {case} has no public model identity")
        requests = {
            request.get("role"): request
            for request in run.get("requests", [])
            if isinstance(request, dict) and isinstance(request.get("role"), str)
        }
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
                request = requests.get(role)
                if not isinstance(request, dict) or request.get("outcome") != "success":
                    raise ReportError(
                        f"constructed run {case} lacks successful symmetric role {role}"
                    )
                ttft = request.get("ttft_ns")
                if not isinstance(ttft, int) or ttft < 0:
                    raise ReportError(f"run {case} has invalid symmetric TTFT for {role}")
                ranked.append((ttft, role))
            ranked.sort()
            for rank, (ttft, role) in enumerate(ranked, start=1):
                key = (model, case, profile, name, roles, rank)
                values[key].append((sample, ttft, str(run["_source"])))
                roles_at_rank[key][role] += 1

    rows: list[dict[str, Any]] = []
    for key, samples in sorted(values.items()):
        model, case, profile, name, roles, rank = key
        category, description = _case_metadata(case, runs)
        rows.append(
            {
                "model": model,
                "category": category,
                "description": description,
                "case": case,
                "profile_label": profile,
                "symmetric_group": name,
                "request_roles": list(roles),
                "rank": rank,
                "role_at_rank_counts": dict(sorted(roles_at_rank[key].items())),
                "raw_directory": str(Path(samples[0][2]).parent),
                **_stats(samples),
            }
        )
    return rows


def _canonical_observations(
    groups: Sequence[dict[str, Any]],
    symmetric_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return comparable observations without assigning meaning to symmetric role labels."""

    symmetric_roles = {
        (row["model"], row["case"], row["profile_label"], role)
        for row in symmetric_rows
        for role in row["request_roles"]
    }
    observations: list[dict[str, Any]] = []
    for row in groups:
        role = row["request_role"]
        key = (row["model"], row["case"], row["profile_label"], role)
        if key in symmetric_roles:
            continue
        observations.append(
            {
                **row,
                "observation_kind": "role",
                "observation_label": role,
                "symmetric_group": None,
                "request_roles": None,
                "rank": None,
            }
        )
    for row in symmetric_rows:
        observations.append(
            {
                **row,
                "observation_kind": "symmetric_rank",
                "observation_label": (
                    f"{row['symmetric_group']} rank {row['rank']}"
                ),
                "request_role": None,
            }
        )
    return observations


def _observation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    common = (row["model"], row["case"], row["profile_label"])
    if row["observation_kind"] == "role":
        return (*common, "role", row["request_role"])
    return (
        *common,
        "symmetric_rank",
        row["symmetric_group"],
        tuple(sorted(row["request_roles"])),
        row["rank"],
    )


def _comparison_rows(
    groups: Sequence[dict[str, Any]],
    run_roles: dict[tuple[str, str, int], dict[str, int]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["case"]), str(row["request_role"])): row
        for row in groups
    }
    models = sorted({str(row["model"]) for row in groups})
    rows: list[dict[str, Any]] = []
    for model in models:
        for definition in COMPARISONS:
            subject = lookup.get((model, definition.subject.case, definition.subject.role))
            baseline = lookup.get((model, definition.baseline.case, definition.baseline.role))
            common = {
                "model": model,
                "name": definition.name,
                "section": definition.section,
                "label": definition.label,
                "subject": {
                    "case": definition.subject.case,
                    "role": definition.subject.role,
                },
                "baseline": {
                    "case": definition.baseline.case,
                    "role": definition.baseline.role,
                },
            }
            if subject is None or baseline is None:
                rows.append(
                    {
                        **common,
                        "available": False,
                        "missing": [
                            side
                            for side, value in (("subject", subject), ("baseline", baseline))
                            if value is None
                        ],
                    }
                )
                continue

            subject_median = float(subject["median_ttft_ns"])
            baseline_median = float(baseline["median_ttft_ns"])
            method = "independent_group_medians"
            compared_samples = min(int(subject["samples"]), int(baseline["samples"]))
            delta = subject_median - baseline_median
            ratio = subject_median / baseline_median if baseline_median > 0 else None
            paired_deltas: list[int] = []
            paired_ratios: list[float] = []
            if definition.subject.case == definition.baseline.case:
                for (run_model, case, _sample), role_values in sorted(run_roles.items()):
                    if run_model != model or case != definition.subject.case:
                        continue
                    left = role_values.get(definition.subject.role)
                    right = role_values.get(definition.baseline.role)
                    if left is not None and right is not None:
                        paired_deltas.append(left - right)
                        if right > 0:
                            paired_ratios.append(left / right)
                if paired_deltas:
                    method = "paired_within_run"
                    compared_samples = len(paired_deltas)
                    delta = float(statistics.median(paired_deltas))
                    ratio = (
                        float(statistics.median(paired_ratios)) if paired_ratios else None
                    )

            direction = "equal"
            if delta > 0:
                direction = "slower"
            elif delta < 0:
                direction = "faster"
            rows.append(
                {
                    **common,
                    "available": True,
                    "method": method,
                    "compared_samples": compared_samples,
                    "subject_samples": int(subject["samples"]),
                    "baseline_samples": int(baseline["samples"]),
                    "subject_median_ttft_ns": subject_median,
                    "subject_median_ttft_ms": subject_median / 1e6,
                    "baseline_median_ttft_ns": baseline_median,
                    "baseline_median_ttft_ms": baseline_median / 1e6,
                    "delta_ns": delta,
                    "delta_ms": delta / 1e6,
                    "ratio": ratio,
                    "speedup": 1.0 / ratio if ratio is not None and ratio > 0 else None,
                    "direction": direction,
                    "paired_delta_ns": paired_deltas,
                    "paired_ratio": paired_ratios,
                }
            )
    return rows


def _cross_campaign_rows(
    current_groups: Sequence[dict[str, Any]],
    baseline_groups: Sequence[dict[str, Any]],
    current_symmetric: Sequence[dict[str, Any]],
    baseline_symmetric: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = _canonical_observations(current_groups, current_symmetric)
    baseline = _canonical_observations(baseline_groups, baseline_symmetric)
    baseline_lookup = {_observation_key(row): row for row in baseline}
    rows: list[dict[str, Any]] = []
    for row in current:
        previous = baseline_lookup.get(_observation_key(row))
        if previous is None:
            continue
        current_median = float(row["median_ttft_ns"])
        baseline_median = float(previous["median_ttft_ns"])
        delta = current_median - baseline_median
        ratio = current_median / baseline_median if baseline_median > 0 else None
        rows.append(
            {
                "model": row["model"],
                "category": row["category"],
                "case": row["case"],
                "profile_label": row["profile_label"],
                "request_role": row["request_role"],
                "observation_kind": row["observation_kind"],
                "observation_label": row["observation_label"],
                "symmetric_group": row["symmetric_group"],
                "request_roles": row["request_roles"],
                "rank": row["rank"],
                "current_samples": row["samples"],
                "baseline_samples": previous["samples"],
                "current_median_ttft_ns": current_median,
                "current_median_ttft_ms": current_median / 1e6,
                "baseline_median_ttft_ns": baseline_median,
                "baseline_median_ttft_ms": baseline_median / 1e6,
                "delta_ns": delta,
                "delta_ms": delta / 1e6,
                "ratio": ratio,
            }
        )
    return rows


def _validate_baseline(current: CampaignData, baseline: CampaignData) -> None:
    for field in ("model_profile", "kv_dtype"):
        if current.manifest.get(field) != baseline.manifest.get(field):
            raise ReportError(
                f"baseline {field} mismatch: "
                f"{baseline.manifest.get(field)!r} != {current.manifest.get(field)!r}"
            )


def summarize_campaign(
    campaign: CampaignData, baseline: CampaignData | None = None
) -> dict[str, Any]:
    groups_map, run_roles, rejected = _request_maps(campaign.runs)
    expected_by_case = Counter(plan.case for plan in campaign.plans)
    groups: list[dict[str, Any]] = []
    for (model, case, profile, role), samples in sorted(groups_map.items()):
        category, description = _case_metadata(case, campaign.runs)
        raw_dir = str(Path(samples[0][2]).parent)
        groups.append(
            {
                "model": model,
                "category": category,
                "description": description,
                "case": case,
                "profile_label": profile,
                "request_role": role,
                "expected_samples": expected_by_case[case],
                "raw_directory": raw_dir,
                **_stats(samples),
            }
        )

    symmetric_rows = _symmetric_rows(campaign.runs)
    canonical_observations = _canonical_observations(groups, symmetric_rows)

    boundary_rows: list[dict[str, Any]] = []
    for (model, case, profile, role), counts in sorted(rejected.items()):
        category, description = _case_metadata(case, campaign.runs)
        for (status, code), count in sorted(counts.items(), key=lambda item: repr(item[0])):
            boundary_rows.append(
                {
                    "model": model,
                    "category": category,
                    "description": description,
                    "case": case,
                    "profile_label": profile,
                    "request_role": role,
                    "http_status": status,
                    "error_code": code,
                    "samples": count,
                    "expected_samples": expected_by_case[case],
                }
            )

    missing = []
    for plan in campaign.plans:
        if plan.raw.is_file():
            continue
        missing_row = {
            "case": plan.case,
            "profile_label": plan.profile,
            "sample": plan.sample,
            "raw": _relative(campaign.root, plan.raw),
            "progress": _relative(campaign.root, plan.progress),
            "serve_log": _relative(campaign.root, plan.serve_log),
        }
        if plan.request_log_jsonl is not None:
            missing_row["request_log_jsonl"] = _relative(
                campaign.root, plan.request_log_jsonl
            )
        missing.append(missing_row)
    not_constructed = [
        {
            "case": run.get("case"),
            "profile_label": run.get("profile_label"),
            "sample": run.get("_sample"),
            "status": run.get("status"),
            "failed_conditions": run.get("failed_conditions", []),
            "raw": run.get("_source"),
            "progress": run.get("_progress"),
            "serve_log": run.get("_serve_log"),
            "request_log_jsonl": run.get("_request_log_jsonl"),
        }
        for run in campaign.runs
        if run.get("constructed") is not True
    ]
    status_counts = Counter(str(run.get("status")) for run in campaign.runs)
    constructed_by_case = Counter(
        str(run.get("case")) for run in campaign.runs if run.get("constructed") is True
    )
    complete_cases = sum(
        constructed_by_case[case] == count for case, count in expected_by_case.items()
    )
    coverage = {
        "campaign_status": campaign.manifest.get("status"),
        "planned_runs": len(campaign.plans),
        "present_artifacts": sum(plan.raw.is_file() for plan in campaign.plans),
        "valid_artifacts": len(campaign.runs),
        "constructed_runs": sum(run.get("constructed") is True for run in campaign.runs),
        "not_constructed_runs": len(not_constructed),
        "missing_artifacts": len(missing),
        "invalid_artifacts": sum(
            error.get("kind") == "run" for error in campaign.artifact_errors
        ),
        "auxiliary_artifact_errors": sum(
            error.get("kind") != "run" for error in campaign.artifact_errors
        ),
        "failure_records": len(campaign.failures),
        "planned_cases": len(expected_by_case),
        "complete_cases": complete_cases,
        "incomplete_cases": len(expected_by_case) - complete_cases,
    }

    variability = sorted(
        (
            {
                "model": row["model"],
                "category": row["category"],
                "case": row["case"],
                "profile_label": row["profile_label"],
                "request_role": row["request_role"],
                "observation_kind": row["observation_kind"],
                "observation_label": row["observation_label"],
                "symmetric_group": row["symmetric_group"],
                "request_roles": row["request_roles"],
                "rank": row["rank"],
                "samples": row["samples"],
                "median_ttft_ms": row["median_ttft_ms"],
                "min_ttft_ms": row["min_ttft_ms"],
                "max_ttft_ms": row["max_ttft_ms"],
                "relative_span": row["relative_span"],
                "raw_directory": row["raw_directory"],
            }
            for row in canonical_observations
            if row["samples"] >= 2 and row["relative_span"] is not None
        ),
        key=lambda row: (
            -float(row["relative_span"]),
            str(row["case"]),
            str(row["observation_label"]),
        ),
    )

    result: dict[str, Any] = {
        "artifact_type": "ninfer_serve_ttft_summary",
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign": {
            "root": str(campaign.root),
            "name": campaign.manifest.get("campaign"),
            "model_profile": campaign.manifest.get("model_profile"),
            "kv_dtype": campaign.manifest.get("kv_dtype"),
            "samples": campaign.manifest.get("samples"),
            "created_at": campaign.manifest.get("created_at"),
            "completed_at": campaign.manifest.get("completed_at"),
        },
        "coverage": coverage,
        "run_status_counts": dict(sorted(status_counts.items())),
        "ttft_groups": groups,
        "comparisons": _comparison_rows(groups, run_roles),
        "symmetric_order_statistics": symmetric_rows,
        "boundary_rejections": boundary_rows,
        "variability": variability,
        "missing_runs": missing,
        "not_constructed_runs": not_constructed,
        "failures": campaign.failures,
        "artifact_errors": campaign.artifact_errors,
    }

    if baseline is not None:
        _validate_baseline(campaign, baseline)
        baseline_summary = summarize_campaign(baseline)
        result["baseline"] = {
            "root": str(baseline.root),
            "name": baseline.manifest.get("campaign"),
            "status": baseline.manifest.get("status"),
            "created_at": baseline.manifest.get("created_at"),
            "completed_at": baseline.manifest.get("completed_at"),
        }
        result["cross_campaign_comparisons"] = _cross_campaign_rows(
            groups,
            baseline_summary["ttft_groups"],
            symmetric_rows,
            baseline_summary["symmetric_order_statistics"],
        )
    else:
        result["cross_campaign_comparisons"] = []

    return result
