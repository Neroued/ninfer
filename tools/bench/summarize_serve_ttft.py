#!/usr/bin/env python3
"""Summarize Serve TTFT run JSON without inferring internal cache actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.ttft.report import load_runs, summarize, write_csv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    summary = summarize(load_runs(args.inputs))
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, summary)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
