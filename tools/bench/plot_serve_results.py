#!/usr/bin/env python3
"""Plot the measured points in a run_serve_concurrency summary."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def report_text(reports, summary, output):
    lines = [
        "# Serve performance report",
        "",
        f"[Measured workload summary]({os.path.relpath(summary.with_suffix('.md'), output)})",
        "",
        "Step times use native counters at the actual batch size. GPU intervals are shown "
        "per rank and can overlap; their sum is not end-to-end latency.",
        "",
    ]
    for point in reports:
        lines += [
            f"## {point['target']} / {point['suite']} / C={point['concurrency']}",
            "",
            f"Requests: {point['request_count']}; completion tokens: {point['totals']['completion_tokens']:,}; "
            f"reasoning tokens: {point['reasoning_tokens']:,}; outcomes: `{point['outcomes']}`.",
            "",
        ]
        batches = point["measurement"]["decode_by_batch"]
        lines += [
            table(
                [
                    "Actual batch",
                    "Steps",
                    "Result ms",
                    "Resolution ms",
                    "Retirement ms",
                    "Total ms",
                ],
                [
                    [
                        row["batch"],
                        row["units"],
                        *(
                            f"{row['stage_mean_ms'][key]:.3f}"
                            for key in ("result", "resolution", "retirement")
                        ),
                        f"{row['mean_ms']:.3f}",
                    ]
                    for row in batches
                    if row["units"]
                ],
            ),
            "",
        ]
        rows = []
        for rank in point["measurement"]["ranks"]:
            for kind in ("prefill", "decode"):
                work = rank["attention_work"][kind]
                rows.append(
                    [
                        rank["rank"],
                        kind,
                        work["rows"],
                        work["valid_positions"],
                        work["scheduled_positions"],
                    ]
                )
        lines += [
            table(
                [
                    "Rank",
                    "Attention work",
                    "Rows",
                    "Valid positions",
                    "Scheduled positions",
                ],
                rows,
            ),
            "",
        ]
        rows = []
        for rank in point["measurement"]["ranks"]:
            for phase, interval in rank["gpu_intervals"].items():
                if not interval["count"]:
                    continue
                rows.append(
                    [
                        rank["rank"],
                        phase,
                        interval["count"],
                        f"{interval['total_ns']/1e6:.2f}",
                        f"{interval['total_ns']/1e6/interval['count']:.3f}",
                    ]
                )
        lines += [
            table(["Rank", "GPU interval", "Samples", "Total ms", "Mean ms"], rows),
            "",
        ]
        if point["repetitive_suffix_requests"]:
            lines += [
                "Repeated suffix flags (retained in the measurements):",
                "",
                *(
                    f"- {row['fixture']}, seed {row['seed']}: a {row['repeated_suffix_characters']}-character suffix repeated at least four times."
                    for row in point["repetitive_suffix_requests"]
                ),
                "",
            ]
    lines += ["## Charts", ""]
    for path in sorted(output.glob("*.svg")):
        lines += [f"![{path.stem}]({path.name})", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads(args.summary.read_text())["points"]
    args.output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for report in reports:
        groups[
            (report["target"], report["kv_dtype"], report["speculative_mode"])
        ].append(report)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    for index, (key, points) in enumerate(groups.items()):
        title = " / ".join(key)

        def save(fig, name):
            fig.suptitle(title)
            for axis in fig.axes:
                axis.grid(axis="y", alpha=0.2)
            for extension in ("svg", "png"):
                fig.savefig(args.output / f"{index}-{name}.{extension}", dpi=160)
            plt.close(fig)

        context = [
            phase
            for point in points
            if point["suite"] == "context-profile"
            for phase in point["phase_summary"]
        ]
        if context:
            context.sort(key=lambda row: row["prompt_tokens_mean"])
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
            for axis, (metric, label) in zip(
                axes,
                (
                    ("prefill_tok_s", "Prefill tokens/s"),
                    ("prefill_chunk_ms", "Prefill chunk ms"),
                    ("decode_step_ms", "Decode step ms (B=1)"),
                ),
            ):
                measured = [row for row in context if row[f"{metric}_mean"] is not None]
                x = [row["prompt_tokens_mean"] for row in measured]
                axis.errorbar(
                    x,
                    [row[f"{metric}_mean"] for row in measured],
                    yerr=[row[f"{metric}_stddev"] for row in measured],
                    fmt="o-",
                    capsize=3,
                )
                axis.set(xscale="log", xlabel="Prompt tokens", ylabel=label)
                axis.set_xticks(x, [f"{value / 1024:.1f}K" for value in x])
            save(fig, "context")

        concurrent = [point for point in points if point["suite"] != "context-profile"]
        if concurrent:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
            for suite, label in (
                ("decode-saturation", "Saturation"),
                ("corpus-makespan", "Corpus"),
            ):
                selected = sorted(
                    (p for p in concurrent if p["suite"] == suite),
                    key=lambda p: p["concurrency"],
                )
                if not selected:
                    continue
                x = [p["concurrency"] for p in selected]
                steps = [
                    next(
                        (
                            row["mean_ms"]
                            for row in p["measurement"]["decode_by_batch"]
                            if row["batch"] == p["concurrency"]
                        ),
                        None,
                    )
                    for p in selected
                ]
                axes[0].plot(x, steps, "o-", label=label)
                axis = axes[1] if suite == "decode-saturation" else axes[2]
                rates = [
                    (
                        p["metrics"]["steady"]["decode_tokens_per_second"]
                        if suite == "decode-saturation"
                        else p["metrics"]["decode_tokens_per_second"]
                    )
                    for p in selected
                ]
                axis.plot(x, rates, "o-")
            axes[0].set_ylabel("Actual full-batch step ms")
            axes[0].legend()
            axes[1].set_ylabel("Steady decode tokens/s")
            axes[2].set_ylabel("Corpus decode tokens/s")
            for axis in axes:
                axis.set_xlabel("Client concurrency")
                axis.set_xticks(sorted({p["concurrency"] for p in concurrent}))
            save(fig, "concurrency")

        phases = [
            phase
            for point in points
            if point["suite"] == "corpus-makespan" and point["concurrency"] == 1
            for phase in point["phase_summary"]
            if phase["section"] in ("scenario_category", "long_decode")
            and phase["decode_tok_s_mean"] is not None
        ]
        if phases:
            fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
            labels = [
                (
                    phase["group"]
                    if phase["section"] == "scenario_category"
                    else f"AIME {phase['fixture'][-2:]}"
                )
                for phase in phases
            ]
            axis.bar(
                labels,
                [phase["decode_tok_s_mean"] for phase in phases],
                yerr=[phase["decode_tok_s_stddev"] for phase in phases],
                capsize=3,
            )
            axis.set_ylabel("Single-request decode tokens/s (mean ± sample SD)")
            save(fig, "scenarios")
    report_text(reports, args.summary, args.output)


if __name__ == "__main__":
    main()
