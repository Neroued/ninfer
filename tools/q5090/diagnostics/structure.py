"""Compare two canonical q5090 structural dumps exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            missing = sorted(set(a) - set(b))
            extra = sorted(set(b) - set(a))
            return f"{path}: keys {missing} missing, {extra} extra"
        for key in sorted(a):
            if (diff := first_diff(a[key], b[key], f"{path}.{key}")) is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            if (diff := first_diff(left, right, f"{path}[{index}]")) is not None:
                return diff
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    left = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    right = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    if (diff := first_diff(left, right)) is not None:
        raise SystemExit(f"structural dumps differ: {diff}")
    print(f"structural dumps identical: {args.reference} == {args.candidate}")


if __name__ == "__main__":
    main()
