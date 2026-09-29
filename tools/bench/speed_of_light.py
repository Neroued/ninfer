"""Estimate NInfer inference roof fractions from a ninfer_bench JSON report.

Inspired by llm.q's speed-of-light accounting (IST-DASLab/llmq,
src/utilities/sol.cpp). This tool uses NInfer's actual text artifact bindings and
Engine phase times; it deliberately reports projection compute and decode weight
traffic separately because they have different hardware limits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject


def _positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _projection_geometry(
    config: dict,
) -> tuple[list[tuple[str, int]], list[list[list[str]]]]:
    """Return mandatory text projections and one selectable group per MoE layer."""
    h = config["hidden_size"]
    projections = []
    expert_groups = []
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"text/layers/{layer}/"
        if kind == "full_attention":
            q = config["num_attention_heads"] * config["head_dim"]
            k = config["num_key_value_heads"] * config["head_dim"]
            projections += [
                (prefix + "attention/" + role, rows * h)
                for role, rows in (
                    ("query", q),
                    ("gate", q),
                    ("key", k),
                    ("value", k),
                )
            ]
            projections.append((prefix + "attention/output", h * q))
        elif kind == "linear_attention":
            kg = (
                config["linear_num_key_heads"] * config["linear_key_head_dim"]
            )
            vg = (
                config["linear_num_value_heads"]
                * config["linear_value_head_dim"]
            )
            nv = config["linear_num_value_heads"]
            projections += [
                (prefix + "gdn/" + role, rows * h)
                for role, rows in (
                    ("a_projection", nv),
                    ("b_projection", nv),
                    ("query", kg),
                    ("key", kg),
                    ("value", vg),
                    ("z", vg),
                )
            ]
            projections.append((prefix + "gdn/output", h * vg))
        else:
            raise ValueError(f"unsupported layer type {kind!r}")

        if "num_experts" not in config:
            intermediate = config["intermediate_size"]
            projections += [
                (prefix + "mlp/" + role, h * intermediate)
                for role in ("gate", "up", "down")
            ]
        else:
            moe = prefix + "moe/"
            projections += [
                (moe + "router", h * config["num_experts"]),
                (moe + "shared_score", h),
            ]
            shared = config["shared_expert_intermediate_size"]
            projections += [
                (moe + "shared/" + role, h * shared)
                for role in ("gate", "up", "down")
            ]
            experts = []
            for expert in range(config["num_experts"]):
                names = [
                    moe + f"experts/{expert}/" + role
                    for role in ("gate", "up", "down")
                ]
                experts.append(names)
            expert_groups.append(experts)
            for role in ("gate", "up", "down"):
                projections.append((
                    moe + f"experts/0/{role}",
                    h
                    * config["moe_intermediate_size"]
                    * config["num_experts_per_tok"],
                ))
    return projections, expert_groups


def _binding_bytes(artifact: Artifact, name: str) -> float:
    binding = artifact.directory.bindings[name]
    parts = binding.get("parts")
    if parts is None:
        parts = [{"object": binding["object"], "range": None}]
    total = 0.0
    for part in parts:
        obj = artifact.by_id[part["object"]]
        if not isinstance(obj, TensorObject):
            raise ValueError(f"{name}: binding is not a tensor")
        elements = math.prod(obj.shape)
        count = (
            elements
            if part["range"] is None
            else part["range"][1] - part["range"][0]
        )
        total += obj.bytes * count / elements
    return total


def _model_work(artifact: Artifact) -> tuple[int, int, float]:
    config = artifact.directory.components["text"]["config"]
    projections, expert_groups = _projection_geometry(config)
    elements = sum(count for _, count in projections)
    expert_names = {
        name for group in expert_groups for expert in group for name in expert
    }
    weight_bytes = sum(
        _binding_bytes(artifact, name)
        for name, _ in projections
        if name not in expert_names
    )
    for experts in expert_groups:
        # Routing is input dependent. The smallest selectable experts give an
        # optimistic traffic estimate even when formats differ across experts.
        selectable = sorted(
            sum(_binding_bytes(artifact, name) for name in expert)
            for expert in experts
        )
        weight_bytes += sum(selectable[: config["num_experts_per_tok"]])
    head_elements = config["hidden_size"] * config["vocab_size"]
    head_bytes = _binding_bytes(artifact, "text/output_head")
    return 2 * elements, 2 * head_elements, weight_bytes + head_bytes


def estimate(
    report: dict,
    artifact: Artifact,
    peak_tflops: float,
    hbm_gbps: float | None,
) -> dict:
    if report.get("artifact_type") != "ninfer_bench_report":
        raise ValueError("input must be a ninfer_bench JSON report")
    if report.get("schema_version") != 15:
        raise ValueError(
            "speed-of-light accounting requires ninfer_bench schema v15"
        )
    if (
        report["load"]["architecture"]
        != artifact.directory.components["text"]["config"]["architectures"][0]
    ):
        raise ValueError("report and artifact architectures differ")
    if report["config"]["speculative_backend"] != "none":
        raise ValueError(
            "speed-of-light accounting currently supports --spec none"
        )
    _positive(peak_tflops, "peak_tflops")
    if hbm_gbps is not None:
        _positive(hbm_gbps, "hbm_gbps")
    body_flops, head_flops, active_weight_bytes = _model_work(artifact)
    kv_capacity = report["memory"]["kv_capacity"]
    kv_payload_bytes = report["memory"]["kv_payload_bytes"]
    if kv_capacity <= 0 or kv_payload_bytes < 0:
        raise ValueError("benchmark KV capacity and payload must be valid")
    kv_bytes_per_token = kv_payload_bytes / kv_capacity
    rows = []
    for test in report["tests"]:
        if test["kind"] not in ("pp", "tg", "pp+tg"):
            raise ValueError(f"unsupported benchmark kind {test['kind']!r}")
        phases = {}
        for phase, tokens, seconds in (
            (
                "prefill",
                test["n_prompt"] if test["kind"] != "tg" else 0,
                test["prefill_seconds_mean"],
            ),
            (
                "decode",
                test["n_gen"] if test["kind"] != "pp" else 0,
                test["decode_seconds_mean"],
            ),
        ):
            if tokens <= 0:
                continue
            _positive(seconds, f"{test['label']} {phase} seconds")
            projection_flops = tokens * body_flops + (
                head_flops if phase == "prefill" else tokens * head_flops
            )
            compute_floor = projection_flops / (peak_tflops * 1e12)
            entry = {
                "tokens": tokens,
                "measured_seconds": seconds,
                "projection_flops": projection_flops,
                "compute_floor_seconds": compute_floor,
                "compute_roof_fraction": compute_floor / seconds,
            }
            if phase == "decode" and hbm_gbps is not None:
                traffic_floor = (
                    tokens * active_weight_bytes / (hbm_gbps * 1e9)
                )
                prompt = test["n_prompt"] if test["kind"] == "pp+tg" else 1
                attention_positions = (
                    tokens * prompt + tokens * (tokens + 1) // 2
                )
                kv_read_bytes = attention_positions * kv_bytes_per_token
                entry.update({
                    "estimated_weight_bytes_per_token": active_weight_bytes,
                    "weight_traffic_seconds": traffic_floor,
                    "weight_roof_fraction": traffic_floor / seconds,
                    "estimated_kv_read_bytes": kv_read_bytes,
                    "kv_traffic_seconds": kv_read_bytes / (hbm_gbps * 1e9),
                    "kv_roof_fraction": kv_read_bytes
                    / (hbm_gbps * 1e9 * seconds),
                })
            phases[phase] = entry
        rows.append({"label": test["label"], "phases": phases})
    return {
        "model": report["load"]["name"],
        "gpu": report["environment"]["gpu_name"],
        "artifact_id": artifact.artifact_id.hex(),
        "peak_tflops": peak_tflops,
        "hbm_gbps": hbm_gbps,
        "body_projection_flops_per_token": body_flops,
        "output_head_flops_per_call": head_flops,
        "estimated_weight_bytes_per_token": active_weight_bytes,
        "estimated_kv_bytes_per_context_token": kv_bytes_per_token,
        "tests": rows,
    }


_ROOF_KEYS = {
    "compute": ("compute_floor_seconds", "compute_roof_fraction"),
    "weight": ("weight_traffic_seconds", "weight_roof_fraction"),
    "kv": ("kv_traffic_seconds", "kv_roof_fraction"),
}


def compare(baseline: dict, current: dict) -> dict:
    """Report how much of the distance to each roof a change closed."""
    if baseline["model"] != current["model"]:
        raise ValueError("baseline and current runs use different models")
    if (
        baseline["peak_tflops"] != current["peak_tflops"]
        or baseline["hbm_gbps"] != current["hbm_gbps"]
    ):
        raise ValueError(
            "baseline and current runs used different ceilings; the roof "
            "fractions are not comparable"
        )
    baseline_phases = {
        (row["label"], phase): entry
        for row in baseline["tests"]
        for phase, entry in row["phases"].items()
    }
    current_phases = {
        (row["label"], phase): entry
        for row in current["tests"]
        for phase, entry in row["phases"].items()
    }
    unmatched = sorted(set(baseline_phases) ^ set(current_phases))
    rows = []
    for current_row in current["tests"]:
        for phase, after in current_row["phases"].items():
            label = current_row["label"]
            before = baseline_phases.get((label, phase))
            if before is None:
                continue
            if before["tokens"] != after["tokens"]:
                raise ValueError(
                    f"{label} {phase} token counts differ between the runs; "
                    "not comparable"
                )
            measured_before = before["measured_seconds"]
            measured_after = after["measured_seconds"]
            row = {
                "label": label,
                "phase": phase,
                "tokens": before["tokens"],
                "measured_seconds_before": measured_before,
                "measured_seconds_after": measured_after,
                "measured_change_fraction": (
                    measured_after - measured_before
                )
                / measured_before,
            }
            for roof, (floor_key, fraction_key) in _ROOF_KEYS.items():
                if fraction_key not in before or fraction_key not in after:
                    continue
                gap_before = measured_before - before[floor_key]
                gap_after = measured_after - after[floor_key]
                row[roof] = {
                    "fraction_before": before[fraction_key],
                    "fraction_after": after[fraction_key],
                    "gap_before_seconds": gap_before,
                    "gap_after_seconds": gap_after,
                    "gap_closed_fraction": (
                        None
                        if gap_before <= 0
                        else (gap_before - gap_after) / gap_before
                    ),
                }
            rows.append(row)
    return {
        "model": current["model"],
        "gpu": current["gpu"],
        "peak_tflops": current["peak_tflops"],
        "hbm_gbps": current["hbm_gbps"],
        "baseline_artifact_id": baseline.get("artifact_id"),
        "artifact_id": current.get("artifact_id"),
        "rows": rows,
        "unmatched": [f"{label} {phase}" for label, phase in unmatched],
    }


def _format_comparison(comparison: dict) -> str:
    hbm = comparison["hbm_gbps"]
    hbm_text = f"{hbm:g} GB/s" if hbm is not None else "n/a"
    lines = [
        f"{comparison['model']} on {comparison['gpu']} "
        f"(ceilings {comparison['peak_tflops']:g} TFLOP/s, {hbm_text})",
        (
            "roof change vs baseline "
            f"({comparison['baseline_artifact_id'] or 'unknown'} -> "
            f"{comparison['artifact_id'] or 'unknown'})"
        ),
    ]
    for row in comparison["rows"]:
        lines.append(
            f"{row['label']} {row['phase']}: "
            f"{row['measured_seconds_before']:.3f}s -> "
            f"{row['measured_seconds_after']:.3f}s "
            f"({100 * row['measured_change_fraction']:+.1f}%)"
        )
        for roof in _ROOF_KEYS:
            if roof not in row:
                continue
            value = row[roof]
            if value["gap_closed_fraction"] is None:
                closed = "n/a (baseline was at or above the roof)"
            else:
                closed = (
                    f"{100 * value['gap_closed_fraction']:+.1f}% of the "
                    f"{100 * (1 - value['fraction_before']):.1f}% gap "
                    "to the roof closed"
                )
            lines.append(
                f"  {roof} roof: {100 * value['fraction_before']:.1f}% -> "
                f"{100 * value['fraction_after']:.1f}% ({closed})"
            )
    for name in comparison["unmatched"]:
        lines.append(f"not present in both runs: {name}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="ninfer_bench -o json report",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="the same v3 .ninfer artifact used for the report",
    )
    parser.add_argument(
        "--peak-tflops",
        required=True,
        type=float,
        help="optimistic tensor-core compute ceiling in TFLOP/s",
    )
    parser.add_argument(
        "--hbm-gbps",
        type=float,
        help="sustained weight-stream bandwidth in GB/s",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable results"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="previous --json output from before the change; report the roof "
        "fraction change and how much of the roof gap the change closed",
    )
    args = parser.parse_args()
    try:
        with args.report.open(encoding="utf-8") as source:
            report = json.load(source)
        with Artifact(args.artifact) as artifact:
            result = estimate(
                report, artifact, args.peak_tflops, args.hbm_gbps
            )
        if args.baseline is not None:
            with args.baseline.open(encoding="utf-8") as source:
                baseline = json.load(source)
            result["comparison"] = compare(baseline, result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2))
    elif "comparison" in result:
        print(_format_comparison(result["comparison"]))
    else:
        print(f"{result['model']} on {result['gpu']}")
        print(
            "Projection compute roof; decode weight and KV roofs (estimates)"
        )
        for row in result["tests"]:
            for phase, value in row["phases"].items():
                line = (
                    f"{row['label']} {phase}: {100 * value['compute_roof_fraction']:.1f}% "
                    "projection compute roof"
                )
                if "weight_roof_fraction" in value:
                    line += (
                        f", {100 * value['weight_roof_fraction']:.1f}% weight roof"
                        f", {100 * value['kv_roof_fraction']:.1f}% KV roof"
                    )
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
