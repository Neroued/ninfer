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
from tools.bench.ttft.diagnostics import (
    MECHANISM_COUNTERS,
    SCHEDULING_FIELDS,
    attach_generation_diagnostics,
    request_timing_analysis,
    unavailable_diagnostics,
)


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
    *tuple(
        _comparison(
            f"{kind}_state_{label}{turn}_vs_{baseline}", "State working set",
            f"{kind.capitalize()} State {label}{turn} vs {label}{baseline}",
            f"{kind}-state-working-set-shift", f"{label}{turn}",
            f"{kind}-state-working-set-shift", f"{label}{baseline}",
        )
        for kind, labels in (("shared", "def"), ("private", "cd"))
        for label in labels for turn in range(1, 4)
        for baseline in ((0,) if turn == 1 else (0, 1))
    ),
    *tuple(
        _comparison(
            f"state_hot_a{turn}_vs_a1", "State working set", f"Hot prefix a{turn} vs a1",
            "shared-state-hot-prefix", f"a{turn}", "shared-state-hot-prefix", "a1",
        )
        for turn in range(2, 7)
    ),
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
    *tuple(
        _comparison(
            f"session_rotation_55k_round2_context_{index}_vs_round1",
            "resource pressure",
            f"55K rotation context {index} round 2 vs round 1",
            "session-rotation-55k-host",
            f"round-2-context-{index}",
            "session-rotation-55k-host",
            f"round-1-context-{index}",
        )
        for index in range(6)
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
        "OpenAI automatic-prefix reuse vs first",
        "shared-openai-implicit",
        "reuse",
        "shared-openai-implicit",
        "first",
    ),
    _comparison(
        "unmarked_repeat_vs_first",
        "private and session reuse",
        "Repeated unmarked prompt vs first",
        "unmarked-repeat-after-interference",
        "second-observation",
        "unmarked-repeat-after-interference",
        "first-observation",
    ),
    _comparison(
        "unmarked_reuse_vs_first",
        "private and session reuse",
        "Unmarked repeat after interference vs first",
        "unmarked-repeat-after-interference",
        "reuse",
        "unmarked-repeat-after-interference",
        "first-observation",
    ),
    _comparison(
        "shared_tools_second_vs_first",
        "shared prefix",
        "Stable tool-prefix second vs first",
        "shared-tools-revisit",
        "stable",
        "shared-tools-revisit",
        "source",
    ),
    _comparison(
        "shared_tools_changed_vs_stable",
        "shared prefix",
        "Changed tool identity vs stable second use",
        "shared-tools-revisit",
        "changed",
        "shared-tools-revisit",
        "stable",
    ),
    _comparison(
        "shared_prefix_competition_final_vs_first",
        "shared prefix",
        "Marked prefix after competition vs first",
        "shared-prefix-competition",
        "a-final",
        "shared-prefix-competition",
        "a-first",
    ),
    _comparison(
        "shared_tool_competition_reuse_vs_first",
        "shared prefix",
        "Tool prefix after competition vs first",
        "shared-tool-prefix-competition",
        "b-reuse",
        "shared-tool-prefix-competition",
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
        "decode-with-short-arrival",
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
        "media-preprocess-revisit",
        "a-warm",
        "media-preprocess-revisit",
        "a-cold",
    ),
    _comparison(
        "media_thrash_final_vs_first",
        "media",
        "A after B/C pressure vs first A",
        "media-preprocess-revisit",
        "a-revisit",
        "media-preprocess-revisit",
        "a-cold",
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
            if request is None:
                raise ReportError(
                    f"{context} lacks symmetric role {role} in group {name}"
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
        diagnostic_error = attach_generation_diagnostics(value, plan.request_log_jsonl)
        if diagnostic_error is not None:
            artifact_errors.append({
                "kind": "request_log", "case": plan.case, "sample": plan.sample,
                "request_log_jsonl": value["_request_log_jsonl"], "error": diagnostic_error,
            })
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
            if any(requests.get(role, {}).get("outcome") != "success" for role in roles):
                continue
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


def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_ns": statistics.mean(ordered) if ordered else None,
        "median_ns": statistics.median(ordered) if ordered else None,
        "p95_ns": ordered[(95 * len(ordered) + 99) // 100 - 1] if ordered else None,
        "max_ns": ordered[-1] if ordered else None,
    }


def _lifecycle_metrics(requests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "requests": len(requests),
        "request_outcomes": dict(Counter(str(r.get("outcome")) for r in requests)),
        **{
            metric: _distribution([r[field] for r in requests if type(r.get(field)) is int])
            for metric, field in (("ttft", "ttft_ns"), ("terminal", "terminal_latency_ns"))
        },
        "output_gap": _distribution([gap for r in requests for gap in r.get("output_gap_ns", [])]),
    }


def _lifecycle_groups(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    samples: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for run in runs:
        if run.get("constructed") is not True:
            continue
        classes = run.get("notes", {}).get("request_classes", {})
        for request in run.get("requests", []):
            role = request["role"]
            kind, label = ("class", classes[role]) if role in classes else ("role", role)
            key = (run["case"], run["profile_label"], kind, label)
            groups[key].append(request)
            samples[key].add(run["_sample"])
    return [
        {"case": case, "profile_label": profile, "group_kind": kind, "group_label": label,
         "samples": len(samples[key]), **_lifecycle_metrics(requests)}
        for key, requests in sorted(groups.items()) for case, profile, kind, label in [key]
    ]


def _phase_metrics(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        by_role = {request["role"]: request for request in run.get("requests", [])}
        for phase, roles in run.get("notes", {}).get("workload_phases", {}).items():
            requests = [by_role[role] for role in roles if role in by_role]
            starts = [r["sent_ns"] for r in requests if type(r.get("sent_ns")) is int]
            ends = [r["ended_ns"] for r in requests if type(r.get("ended_ns")) is int]
            complete = bool(roles) and len(requests) == len(roles)
            timed = complete and len(starts) == len(roles) and len(ends) == len(roles)
            rows.append({
                "case": run["case"], "sample": run["_sample"], "phase": phase,
                "scope": "selected_client_requests", "constructed": run.get("constructed"),
                "missing_roles": [role for role in roles if role not in by_role],
                "duration_ns": max(ends) - min(starts) if timed else None,
                **_lifecycle_metrics(requests),
            })
    return rows


def _arrival_metrics(notes: dict[str, Any], requests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Use only scheduled roles and the client's monotonic clock, excluding setup requests."""
    arrivals = notes.get("arrivals") or []
    if not arrivals:
        return {}
    roles = {arrival["role"] for arrival in arrivals}
    scheduled = [r for r in requests if r.get("role") in roles]
    offsets = [a["offset_ns"] for a in arrivals if type(a.get("offset_ns")) is int]
    starts = [r["sent_ns"] for r in scheduled if type(r.get("sent_ns")) is int]
    ends = [r["ended_ns"] for r in scheduled if type(r.get("ended_ns")) is int]
    terminals = [r["terminal_ns"] for r in scheduled if type(r.get("terminal_ns")) is int]
    success = [r for r in scheduled if r.get("outcome") == "success"]
    tokens = [r.get("usage", {}).get("output_tokens") for r in success]
    complete_starts = len(starts) == len(roles)
    complete_ends = len(ends) == len(roles)
    late = [a["lateness_ns"] for a in arrivals if type(a.get("lateness_ns")) is int]
    return {
        "scheduled_requests": len(roles), "scheduled_successful_requests": len(success),
        "scheduled_completed_output_tokens": sum(tokens) if all(type(n) is int for n in tokens) else None,
        "planned_injection_span_ns": max(offsets) - min(offsets) if len(offsets) == len(roles) else None,
        "actual_injection_span_ns": max(starts) - min(starts) if complete_starts else None,
        "max_send_lateness_ns": max(late) if len(late) == len(roles) else None,
        "drain_ns": max(ends) - max(starts) if complete_starts and complete_ends else None,
        "terminal_drain_ns": max(terminals) - max(starts)
                             if complete_starts and len(terminals) == len(roles) else None,
    }


def _contract_rejections(current: CampaignData, baseline: CampaignData) -> list[dict[str, Any]]:
    def contracts(runs: Sequence[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
        result: dict[tuple[str, str], set[str]] = defaultdict(set)
        for run in runs:
            result[(run["case"], run["profile_label"])].add(
                str(run.get("notes", {}).get("measurement_contract", "1")))
        return result
    current_contracts, baseline_contracts = contracts(current.runs), contracts(baseline.runs)
    return [
        {"case": key[0], "profile_label": key[1], "reason": "measurement_contract_mismatch",
         "current_contracts": sorted(current_contracts[key]),
         "baseline_contracts": sorted(baseline_contracts[key])}
        for key in sorted(current_contracts.keys() & baseline_contracts.keys())
        if len(current_contracts[key]) != 1 or current_contracts[key] != baseline_contracts[key]
    ]


def _stream_observations(runs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep failed and unconstructed measurements visible beside successful comparisons."""
    observations: list[dict[str, Any]] = []
    workloads: list[dict[str, Any]] = []
    for run in runs:
        requests = run.get("requests", [])
        notes = run.get("notes", {})
        starts = [r["sent_ns"] for r in requests if isinstance(r.get("sent_ns"), int)]
        ends = [r["ended_ns"] for r in requests if isinstance(r.get("ended_ns"), int)]
        duration = max(ends) - min(starts) if starts and ends else None
        completed_tokens = 0
        successful = 0
        missing_usage = 0
        for request in requests:
            usage = request.get("usage", {})
            tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
            diagnostics = request.get("diagnostics", unavailable_diagnostics("request_log_not_loaded"))
            counts = diagnostics.get("scheduling", {})
            server_result = diagnostics.get("result", {})
            server_tokens = server_result.get("completion_tokens")
            usage_check = (
                "matched" if tokens == server_tokens else "mismatch"
            ) if type(tokens) is int and type(server_tokens) is int else "unavailable"
            gaps = sorted(request.get("output_gap_ns", []))
            if request.get("outcome") == "success":
                successful += 1
                if type(tokens) is int:
                    completed_tokens += tokens
                else:
                    missing_usage += 1
            observations.append({
                "case": run.get("case"),
                "profile_label": run.get("profile_label"),
                "sample": run.get("_sample"),
                "request_role": request.get("role"),
                "request_class": notes.get("request_classes", {}).get(request.get("role")),
                "constructed": run.get("constructed"),
                "outcome": request.get("outcome"),
                "ttft_ns": request.get("ttft_ns"),
                "terminal_latency_ns": request.get("terminal_latency_ns"),
                "transport_duration_ns": request.get("transport_duration_ns"),
                "max_output_gap_ns": request.get("max_output_gap_ns"),
                "median_output_gap_ns": statistics.median(gaps) if gaps else None,
                "p95_output_gap_ns": gaps[(95 * len(gaps) + 99) // 100 - 1] if gaps else None,
                "terminal_tail_ns": request.get("terminal_tail_ns"),
                "unterminated_tail_ns": request.get("unterminated_tail_ns"),
                "output_event_count": request.get("output_event_count"),
                "output_bytes": request.get("output_bytes"),
                "output_tokens": tokens,
                "wire_request_id": request.get("wire_request_id"),
                "response_id": request.get("response_id"),
                "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                "finish_reason": server_result.get("finish_reason"),
                "diagnostic_status": diagnostics["status"],
                "diagnostic_reason": diagnostics.get("reason"),
                "server_instance_id": diagnostics.get("server_instance_id"),
                "service_request_id": diagnostics.get("service_request_id"),
                "engine_request_id": diagnostics.get("engine_request_id"),
                "mechanisms": diagnostics["mechanisms"],
                "usage_check": usage_check,
                **{field: counts.get(field) for field in SCHEDULING_FIELDS},
                "raw": run.get("_source"),
            })
        complete_timing = len(starts) == len(requests) and len(ends) == len(requests)
        measurable = bool(requests) and complete_timing and duration is not None and duration > 0 and not missing_usage
        comparable = notes.get("throughput_comparable", True) is True and run.get("constructed") is True
        input_dependency = notes.get("input_dependency", "unknown")
        matched_input_eligible = comparable and input_dependency == "fixed"
        limitation = notes.get("throughput_limitation")
        if run.get("constructed") is not True:
            limitation = "; ".join(filter(None, (
                limitation, "The requested arrival graph was not constructed.",
            )))
        mechanism_coverage = {}
        for name in MECHANISM_COUNTERS:
            states = [request.get("diagnostics", {}).get("mechanisms", {}).get(name, "unavailable")
                      for request in requests]
            mechanism_coverage[name] = (
                "observed" if "observed" in states
                else "not_observed" if states and all(state == "not_observed" for state in states)
                else "unavailable"
            )
        for name, state in notes.get("mechanism_observations", {}).items():
            if name not in mechanism_coverage:
                mechanism_coverage[name] = state if state in (
                    "observed", "not_observed", "unavailable",
                ) else "unavailable"
        mechanism_coverage.update(run.get("scheduling_observations", {}).get("mechanisms", {}))
        by_role = {request.get("role"): request for request in requests}
        order_groups = notes.get("expected_engine_order_groups")
        if order_groups:
            ids = [[by_role.get(role, {}).get("diagnostics", {}).get("engine_request_id")
                    for role in roles] for roles in order_groups]
            valid = all(group and all(type(value) is int for value in group) for group in ids)
            mechanism_coverage["expected_engine_order"] = (
                "observed" if all(max(left) < min(right) for left, right in zip(ids, ids[1:]))
                else "not_observed") if valid else "unavailable"
        for name, requirement in notes.get("mechanism_role_requirements", {}).items():
            if name in mechanism_coverage:
                continue
            diagnostic = by_role.get(requirement.get("role"), {}).get("diagnostics", {})
            counter = requirement.get("counter")
            value = diagnostic.get("scheduling", {}).get(counter)
            mechanism_coverage[name] = (
                "observed" if value > 0 else "not_observed"
            ) if (counter in SCHEDULING_FIELDS and diagnostic.get("status") == "available"
                  and type(value) is int) else "unavailable"
        required = notes.get("mechanism_requirements", [])
        required_states = [mechanism_coverage.get(name, "unavailable") for name in required]
        required_status = (
            "not_required" if not required_states
            else "observed" if all(state == "observed" for state in required_states)
            else "unavailable" if "unavailable" in required_states else "not_observed"
        )
        workloads.append({
            "case": run.get("case"),
            "sample": run.get("_sample"),
            "constructed": run.get("constructed"),
            "arrival_mode": notes.get("arrival_mode", "causal"),
            "measurement_contract": str(notes.get("measurement_contract", "1")),
            **_arrival_metrics(notes, requests),
            "throughput_comparable": comparable,
            "input_dependency": input_dependency,
            "matched_input_eligible": matched_input_eligible,
            "throughput_limitation": limitation,
            "mechanism_coverage": mechanism_coverage,
            "mechanism_requirements": required,
            "replay_progress_intervals": run.get("scheduling_observations", {}).get("replay_intervals", []),
            "incomplete_scheduling_requests": run.get("scheduling_observations", {}).get("incomplete_requests", []),
            "required_mechanism_status": required_status,
            "observed_workload": notes.get("observed_workload"),
            "arrivals": notes.get("arrivals"),
            "requests": len(requests),
            "successful_requests": successful,
            "request_outcomes": dict(Counter(str(r.get("outcome")) for r in requests)),
            "missing_success_usage": missing_usage,
            "measured_completed_output_tokens": completed_tokens,
            "duration_ns": duration if complete_timing else None,
            "completed_output_tokens_per_second": (
                completed_tokens * 1e9 / duration if measurable and comparable else None
            ),
            "observed_completed_output_tokens_per_second": (
                completed_tokens * 1e9 / duration if measurable else None
            ),
            "raw": run.get("_source"),
        })
    return observations, workloads


def _request_timing_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"case": run.get("case"), "profile_label": run.get("profile_label"),
         "sample": run.get("_sample"), "request_role": request.get("role"),
         "outcome": request.get("outcome"), "constructed": run.get("constructed"),
         "wire_request_id": request.get("wire_request_id"),
         "engine_request_id": request.get("diagnostics", {}).get("engine_request_id"),
         **request_timing_analysis(request), "raw": run.get("_source")}
        for run in runs for request in run.get("requests", [])
    ]


def summarize_campaign(
    campaign: CampaignData, baseline: CampaignData | None = None
) -> dict[str, Any]:
    groups_map, run_roles, rejected = _request_maps(campaign.runs)
    stream_observations, workload_metrics = _stream_observations(campaign.runs)
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
        "stream_observations": stream_observations,
        "request_lifecycle_groups": _lifecycle_groups(campaign.runs),
        "workload_phase_metrics": _phase_metrics(campaign.runs),
        "request_timing_analysis": _request_timing_rows(campaign.runs),
        "workload_metrics": workload_metrics,
        "global_runtime_observations": [
            {"case": run.get("case"), "sample": run.get("_sample"), "raw": run.get("_source"),
             **run.get("global_diagnostics", {"status": "unavailable", "reason": "request_log_not_loaded"})}
            for run in campaign.runs
        ],
        "mechanism_coverage_counts": dict(Counter(
            row["required_mechanism_status"] for row in workload_metrics
        )),
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
        result["comparison_qualification"] = {
            "status": "conditions_unverified",
            "checked": ["model_profile", "kv_dtype", "measurement_contract"],
            "reason": "Resource geometry, backend, Graph, observation settings and offered inputs "
                      "must be checked before treating these observed TTFT deltas as performance changes.",
        }
        rejected = _contract_rejections(campaign, baseline)
        rejected_keys = {(row["case"], row["profile_label"]) for row in rejected}
        result["cross_campaign_rejected"] = rejected
        result["cross_campaign_comparisons"] = [row for row in _cross_campaign_rows(
            groups,
            baseline_summary["ttft_groups"],
            symmetric_rows,
            baseline_summary["symmetric_order_statistics"],
        ) if (row["case"], row["profile_label"]) not in rejected_keys]
    else:
        result["cross_campaign_rejected"] = []
        result["cross_campaign_comparisons"] = []

    return result
