"""Replay Codacus's pinned playground requests against a running --jev server.

Standard-library HTTP client only: no model loading, downloads or inference runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/decision/codacus.json"
# NInfer-owned semantic check, derived from the arena preset's explicit rules.
# Codacus does not publish golden answers for the presets.
ARENA_EXPECTED = {
    "movement": "right", "turn_dir": "right", "turn_angle": 33,
    "fire": False, "stance": "stand",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def typed_equal(left: object, right: object) -> bool:
    # JSON booleans are not numeric 0/1, despite Python equality.
    return left == right and (isinstance(left, bool) == isinstance(right, bool))


def in_domain(value: object, spec: dict) -> bool:
    if "enum" in spec or "choices" in spec:
        return isinstance(value, str) and value in spec.get("enum", spec.get("choices", []))
    if spec["type"] == "boolean":
        return type(value) is bool
    if not finite(value) or not spec["minimum"] <= value <= spec["maximum"]:
        return False
    if spec["type"] == "integer":
        return type(value) is int
    step = spec.get("step", spec.get("multipleOf"))
    index = (value - spec["minimum"]) / step
    return abs(index - round(index)) <= 1e-7


def validate_response(request: dict, response: dict) -> None:
    require(response.get("object") == "decision", "not a NInfer decision response")
    schema = request["schema"].get("properties", request["schema"])
    results = response.get("results")
    require(isinstance(results, list) and len(results) == len(request["contexts"]),
            "result count does not match contexts")
    for index, result in enumerate(results):
        require(set(result["decision"]) == set(schema), f"context {index}: wrong decision fields")
        require(set(result["fields"]) == set(schema), f"context {index}: wrong diagnostic fields")
        for name, spec in schema.items():
            value = result["decision"][name]
            field = result["fields"][name]
            require(in_domain(value, spec), f"context {index}/{name}: invalid typed value {value!r}")
            require(typed_equal(field["value"], value), f"{name}: inconsistent field value")
            probability = field["probability"]
            require(finite(probability) and 0 <= probability <= 1, f"{name}: invalid probability")
            require(type(field["tree"]) is bool, f"{name}: invalid tree flag")
            require(type(field["scored_nodes"]) is int and field["scored_nodes"] >= 0,
                    f"{name}: invalid scored_nodes")
            numeric = spec.get("type") in ("integer", "number")
            require(("interval_p10_p90" in field) == (numeric and field["tree"]),
                    f"{name}: wrong interval contract")
            if "interval_p10_p90" in field:
                interval = field["interval_p10_p90"]
                require(isinstance(interval, list) and len(interval) == 2 and
                        all(in_domain(v, spec) for v in interval) and interval[0] <= interval[1],
                        f"{name}: invalid numeric interval")
    usage = response["usage"]
    for key in ("prompt_tokens", "computed_tokens", "cached_tokens", "context_tokens", "scored_rows"):
        require(type(usage[key]) is int and usage[key] >= 0, f"invalid usage.{key}")
    require(usage["prompt_tokens"] == usage["computed_tokens"] + usage["cached_tokens"],
            "prompt accounting does not reconcile")
    # cached_tokens includes work shared within this request, even with cache_prompt=false.
    for key in ("prefill_ms", "scoring_ms", "total_ms", "per_decision_ms"):
        require(finite(response["timings"][key]) and response["timings"][key] >= 0,
                f"invalid timings.{key}")


def post(base_url: str, payload: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key := os.environ.get("NINFER_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/decision",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            require(response.status == 200, f"unexpected HTTP {response.status}")
            require(response.headers.get_content_type() == "application/json", "not JSON")
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {error.code}: {detail}") from error


def main() -> int:
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["presets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="server origin, without /v1")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preset", choices=[p["id"] for p in fixtures], default="routing")
    selection.add_argument("--all", action="store_true", help="exercise all eight presets")
    parser.add_argument("--list", action="store_true", help="list fixtures without contacting a server")
    parser.add_argument("--model", help="optional served model alias, not an artifact path")
    parser.add_argument("--mode", choices=("auto", "tree", "greedy"), help="override engine mode")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="repetitions per preset; no speed assertion")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-arena", action="store_true", help="also require five rule-derived arena answers")
    args = parser.parse_args()
    if args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeat and timeout must be positive")
    if args.check_arena and not (args.all or args.preset == "arena"):
        parser.error("--check-arena needs --preset arena or --all")
    if args.list:
        for preset in fixtures:
            payload = preset["request"]
            fields = payload["schema"].get("properties", payload["schema"])
            print(f"{preset['id']}: {len(fields)} fields, {len(payload['contexts'])} context(s)")
        return 0
    selected = [p for p in fixtures if args.all or p["id"] == args.preset]
    semantic_errors = []
    for preset in selected:
        payload = dict(preset["request"])
        if args.model:
            payload["model"] = args.model
        if args.mode:
            payload["mode"] = args.mode
        if args.no_cache:
            payload["cache_prompt"] = False
        elapsed = []
        for run in range(args.repeat):
            start = time.perf_counter()
            response = post(args.base_url, payload, args.timeout)
            elapsed.append((time.perf_counter() - start) * 1000)
            validate_response(payload, response)
            print(json.dumps({"preset": preset["id"], "run": run + 1,
                              "client_ms": elapsed[-1], "response": response}, ensure_ascii=False))
            if args.check_arena and preset["id"] == "arena":
                actual = response["results"][0]["decision"]
                for name, expected in ARENA_EXPECTED.items():
                    if not typed_equal(actual[name], expected):
                        semantic_errors.append(f"run {run + 1} {name}: expected {expected!r}, got {actual[name]!r}")
        print(f"PASS contract {preset['id']}; client ms min/median/max "
              f"{min(elapsed):.1f}/{statistics.median(elapsed):.1f}/{max(elapsed):.1f}", file=sys.stderr)
    if semantic_errors:
        raise ValueError("arena model-quality check failed: " + "; ".join(semantic_errors))
    if args.check_arena:
        print("PASS arena: all five rule-derived decisions; not a general accuracy benchmark", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"FAIL decision smoke: {error}", file=sys.stderr)
        sys.exit(1)
