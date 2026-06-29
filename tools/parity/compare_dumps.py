#!/usr/bin/env python3
"""Compare q5090 v2 structural dumps.

Usage:
  python -m tools.parity.compare_dumps out/conv_dump.v2.json out/ref_dump.v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        ak = set(a)
        bk = set(b)
        if ak != bk:
            missing = sorted(ak - bk)
            extra = sorted(bk - ak)
            parts = []
            if missing:
                parts.append(f"missing in test: {missing[:8]}")
            if extra:
                parts.append(f"extra in test: {extra[:8]}")
            return f"{path}: key mismatch ({'; '.join(parts)})"
        for key in sorted(a):
            diff = _first_diff(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            diff = _first_diff(ai, bi, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def _load(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", help="trusted dump JSON, usually converter output")
    ap.add_argument("test", help="dump JSON to compare, usually Python ref output")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    test_path = Path(args.test)
    ref = _load(ref_path)
    test = _load(test_path)
    diff = _first_diff(ref, test)
    if diff is not None:
        print(f"dumps differ: {diff}")
        raise SystemExit(1)
    print(f"dumps identical: {ref_path} == {test_path}")


if __name__ == "__main__":
    main()
