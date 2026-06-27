#!/usr/bin/env python3
"""Generate the project-authored long_2k prompt text."""

from __future__ import annotations

import argparse
from pathlib import Path


PARAGRAPHS = [
    (
        "This section describes a benchmark review meeting for a transformer inference engine. "
        "The team tracks model loading, prompt prefill, greedy decode, token readback, memory peaks, "
        "fixture identity, and report reproducibility as separate concerns. "
        "Every paragraph is original project text and is intentionally repetitive so the prompt is long "
        "enough to exercise prefill behavior without depending on external documents."
    ),
    (
        "The reviewer asks for concrete evidence instead of broad claims. "
        "The engineer records the command line, git commit, q5090 file identity, prompt token count, "
        "generated token ids, and arena high-water marks. "
        "The report is useful only when another developer can audit the fixture and reproduce the "
        "same input ids from the committed text and tokenizer metadata."
    ),
    (
        "The discussion avoids kernel optimization. "
        "The purpose of this long prompt is to create a stable prefill workload, not to improve throughput. "
        "The generated answer may be inspected by a human, but correctness gates compare token ids, schema "
        "fields, timing boundaries, memory accounting, and deterministic repeat behavior."
    ),
]


def build_text(repeats: int) -> str:
    lines = [
        "Long-context benchmark prompt for M2.8.",
        "",
        "Read the repeated review notes and summarize the operational risks at the end.",
        "",
    ]
    for i in range(repeats):
        lines.append(f"Review note {i + 1:03d}.")
        lines.extend(PARAGRAPHS)
        lines.append("")
    lines.append(
        "Final instruction: summarize the three most important benchmark-readiness risks and name the "
        "evidence that would make each risk auditable."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=36)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_text(args.repeats), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
