#!/usr/bin/env python3
"""Validate a ladder result before it is shared (issue #30).

A shared results log is only useful if every row records the same things, measured the same way.
This refuses a row that is missing the environment that makes it comparable, or whose numbers are
self-contradictory. Run it on a result.json; exit 0 if the row is well formed, 1 otherwise.

    python3 bench/capy_flash/check_result.py result.json
"""

import json
import sys

SCHEMA = "capy-flash-ladder/1"


def check(path):
    errors = []
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return [f"not readable JSON: {e}"]

    if d.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}, got {d.get('schema')!r}")
    for key in ("timestamp", "model", "environment", "results"):
        if not d.get(key):
            errors.append(f"missing {key}")

    env = d.get("environment") or {}
    if not env.get("gpus"):
        errors.append("environment.gpus is empty (nvidia-smi did not run on the bench machine?)")
    if "ram_gib" not in env:
        errors.append("environment.ram_gib missing")
    if "wsl2" not in env:
        errors.append("environment.wsl2 missing (it changes P2P and memory, so it must be stated)")

    results = d.get("results") or []
    if not results:
        errors.append("results is empty")
    seen = set()
    for i, r in enumerate(results):
        where = f"results[{i}] (target {r.get('target_tokens')})"
        size = r.get("target_tokens")
        if size in seen:
            errors.append(f"{where}: duplicate target_tokens")
        seen.add(size)
        pt = r.get("prompt_tokens")
        if pt is None:
            errors.append(f"{where}: prompt_tokens missing (the server never reported a prompt)")
        elif size and (pt < size * 0.5 or pt > size * 1.6):
            errors.append(f"{where}: prompt_tokens {pt} far from target {size}")
        for field in ("prefill_tps_median", "decode_tps_median"):
            v = r.get(field)
            if v is None:
                errors.append(f"{where}: {field} is null (no timings from the server)")
            elif not (0 < v < 1_000_000):
                errors.append(f"{where}: {field}={v} out of range")
    return errors


def main():
    if len(sys.argv) != 2:
        print("usage: check_result.py result.json", file=sys.stderr)
        return 2
    errors = check(sys.argv[1])
    if errors:
        print("INVALID result:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: result is well formed and comparable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
