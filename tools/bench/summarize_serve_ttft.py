#!/usr/bin/env python3
"""Write the standard human- and machine-readable summary for one TTFT campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.ttft.render import write_campaign_summary
from tools.bench.ttft.report import ReportError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="optional compatible campaign used for cross-campaign median comparisons",
    )
    args = parser.parse_args(argv)
    try:
        summary, paths = write_campaign_summary(args.campaign_dir, args.baseline)
    except ReportError as error:
        parser.exit(1, f"cannot summarize TTFT campaign: {error}\n")
    coverage = summary["coverage"]
    print(
        "TTFT summary: "
        f"{coverage['constructed_runs']}/{coverage['planned_runs']} constructed, "
        f"{coverage['failure_records']} failure(s)"
    )
    print(f"report: {paths['markdown']}")
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
