"""Report greedy first divergence from a recorded quantized snapshot.

Token equality is diagnostic only: the Python and C++ numerical paths are
intentionally independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.q5090.ref import RefModel
from tools.q5090.ref.cli import parse_bytes, parse_ids


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def snapshot_tokens(path: Path, case: str, repeat: int) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    for item in report.get("cases", []):
        if item.get("name") == case:
            repeats = item.get("repeats", [])
            if repeat < 0 or repeat >= len(repeats):
                raise ValueError(f"{path}: invalid repeat {repeat} for {case}")
            tokens = repeats[repeat].get("generated_token_ids")
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise ValueError(f"{path}: invalid generated_token_ids for {case}")
            return tokens
    raise ValueError(f"{path}: case {case!r} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--case")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--headroom", default="2GiB")
    parser.add_argument("--json")
    args = parser.parse_args()
    fixture = Path(args.fixture)
    prompt = parse_ids(fixture.read_text(encoding="utf-8"))
    case = args.case or fixture.stem
    expected = snapshot_tokens(Path(args.snapshot), case, args.repeat)
    with RefModel(
        args.weights,
        device=args.device,
        memory_bytes=parse_bytes(args.gpu_memory),
        headroom_bytes=parse_bytes(args.headroom) or 0,
    ) as model, torch.inference_mode():
        actual = model.generate(prompt, len(expected))
        plan = model.weights.plan.summary()
    divergence = first_divergence(actual, expected)
    report = {
        "format": "q5090_greedy_diagnostic_v1",
        "case": case,
        "repeat": args.repeat,
        "expected_tokens": expected,
        "actual_tokens": actual,
        "first_divergence": divergence,
        "memory_plan": plan,
    }
    if divergence is None:
        print(f"no divergence in {len(expected)} tokens")
    else:
        print(
            f"first divergence={divergence} "
            f"python={actual[divergence] if divergence < len(actual) else 'EOF'} "
            f"snapshot={expected[divergence] if divergence < len(expected) else 'EOF'}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
