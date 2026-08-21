"""Six-step cross-branch regression for the content-addressed host KV cache.

Minimal sequence isolating the lane-checkpoint defect fixed in b1cf9801 (credit: vromanov's
report on PR #73): two conversation branches share a long turn-1 prefix; branch B must arrive
via a host-cache content restore, after which re-entering branch A over the device
turn-checkpoint path must still produce branch A's answer. Also exercises the pinned-store
save/restore paths end to end. Talks to a running ninfer-serve started with
--kv-host-cache-mib > 0; greedy sampling throughout. Stdlib only.

Usage: python3 kv_cache_six_step.py --base-url http://127.0.0.1:8080 --model <id> [--filler N]
Exit code 0 when every step returns the branch-correct answer; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def ask(base_url: str, model: str, messages: list[dict], max_tokens: int = 24) -> tuple[str, int]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "enable_thinking": False,
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"], data["usage"]["prompt_tokens"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument(
        "--filler",
        type=int,
        default=1600,
        help="repetitions of the filler sentence in the shared prefix (~11 tokens each)",
    )
    args = parser.parse_args()

    salt = int(time.time())
    prefix = (
        f"Session {salt}. Record 00000: baseline entry.\n"
        + "Audit line item with routine status and no anomalies recorded. " * args.filler
        + "\nIn total 5200 records were audited."
    )
    question_a = "Branch A: how many records were audited in total? Answer with the number only."
    question_b = "Branch B: what is the id of the very first record? Answer with the id only."

    steps: list[tuple[str, str]] = []

    def run(label: str, messages: list[dict], expect: str | None) -> str:
        answer, prompt_tokens = ask(args.base_url, args.model, messages)
        verdict = "ok" if expect is None or expect in answer else f"EXPECTED {expect}"
        steps.append((label, verdict))
        print(f"{label}: prompt={prompt_tokens} answer={answer[:40]!r} [{verdict}]")
        return answer

    reply = run("1 P (cold)", [{"role": "user", "content": prefix}], None)
    conv_a = [
        {"role": "user", "content": prefix},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": question_a},
    ]
    conv_b = [
        {"role": "user", "content": prefix},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": question_b},
    ]
    run("2 conv-A (resident)", conv_a, "5200")
    run("3 short (displace)", [{"role": "user", "content": "Say hi."}], None)
    run("4 conv-B (content restore)", conv_b, "00000")
    run("5 conv-A (content restore)", conv_a, "5200")
    run("6 conv-A (turn checkpoint)", conv_a, "5200")

    failed = [label for label, verdict in steps if verdict != "ok"]
    if failed:
        print(f"FAIL: {', '.join(failed)}")
        return 1
    print("six-step cache regression: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
