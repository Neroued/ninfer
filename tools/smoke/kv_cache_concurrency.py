"""Concurrent-fire battery for the content-addressed host KV cache.

Fires N parallel clients at a running ninfer-serve (started with --max-concurrency >= 2 and
--kv-host-cache-mib > 0) through four phases:

  A. identical cold burst      — N copies of one unseen ~30k-token payload at once; asserts a
                                 single distinct answer and no HTTP errors, reports TTFT spread
                                 (prefill is lane-exclusive, so wall times stack by design);
  B. identical warm burst      — displace, then the same payload again; every request must
                                 answer byte-identically to phase A;
  C. mixed load                — identical group + shared-root group + disjoint group fired
                                 together; per-group answer parity, zero errors;
  D. mid-turn fork burst       — N payloads diverging inside one user message (no completed
                                 turn between them): the documented no-anchor shape; reported
                                 honestly as N full prefills, answers must stay per-fork stable.

Greedy sampling throughout. Stdlib only. Watch the serve log's periodic `host-cache` line while
this runs: `evicted=` climbing with `stored=` pinned at the budget means the working set does
not fit the configured budget.

Usage: python3 kv_cache_concurrency.py --base-url http://HOST:PORT --model <id>
       [--clients 8] [--filler 2700]
Exit 0 when every phase holds parity with zero transport errors; 1 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request


def ask(base_url: str, model: str, messages: list[dict], max_tokens: int = 24):
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
    started = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"], time.time() - started


def fire(base_url: str, model: str, payloads: list[list[dict]]):
    answers: list[str | None] = [None] * len(payloads)
    walls: list[float] = [0.0] * len(payloads)
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures = {
            pool.submit(ask, base_url, model, payload): index
            for index, payload in enumerate(payloads)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                answers[index], walls[index] = future.result()
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
                errors.append(f"client {index}: {error}")
    return answers, walls, errors


def report(phase: str, answers, walls, errors, groups: list[list[int]]) -> bool:
    ok = not errors
    for group in groups:
        distinct = {answers[i] for i in group if answers[i] is not None}
        if len(distinct) != 1:
            ok = False
            print(f"{phase}: group {group} produced {len(distinct)} distinct answers")
    spread = f"walls {min(walls):.2f}s..{max(walls):.2f}s" if walls else "no walls"
    print(f"{phase}: errors={len(errors)} {spread} [{'ok' if ok else 'FAIL'}]")
    for line in errors:
        print(f"  {line}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--filler", type=int, default=2700, help="~11 tokens per repetition")
    args = parser.parse_args()

    salt = int(time.time())
    n = args.clients
    filler = "Audit line item with routine status and no anomalies recorded. "

    def payload(text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    base = (
        f"Session {salt}. Record 00000: baseline entry.\n"
        + filler * args.filler
        + "\nIn total 5200 records were audited. How many records were audited? Digits only."
    )
    passed = True

    answers_a, walls, errors = fire(args.base_url, args.model, [payload(base)] * n)
    passed &= report("A cold identical burst", answers_a, walls, errors, [list(range(n))])
    reference = next((a for a in answers_a if a is not None), None)

    fire(args.base_url, args.model, [payload("Say hi.")])
    answers_b, walls, errors = fire(args.base_url, args.model, [payload(base)] * n)
    passed &= report("B warm identical burst", answers_b, walls, errors, [list(range(n))])
    if reference is not None and any(a is not None and a != reference for a in answers_b):
        print("B: warm answers diverged from cold reference")
        passed = False

    shared_root = f"Shared {salt}. " + filler * args.filler
    half = n // 2
    quarter = max(1, n // 4)
    mixed = (
        [payload(base)] * half
        + [payload(shared_root + f" Branch {i}: reply with the word alpha{i}.") for i in range(quarter)]
        + [payload(f"Disjoint {salt} {i}. " + filler * (args.filler // 2) + " Reply ok.") for i in range(n - half - quarter)]
    )
    groups = [list(range(half))] + [[half + i] for i in range(n - half)]
    answers_c, walls, errors = fire(args.base_url, args.model, mixed)
    passed &= report("C mixed load", answers_c, walls, errors, groups)

    forks = [
        payload(base + f" Mid-turn fork marker {i}: reply with the number {i * 111}.")
        for i in range(n)
    ]
    answers_d, walls, errors = fire(args.base_url, args.model, forks)
    passed &= report("D mid-turn fork burst", answers_d, walls, errors, [[i] for i in range(n)])

    long_gen = f"LongGen {salt}. " + filler * args.filler + " Write a short story about an audit."
    def ask_long(_):
        return ask(args.base_url, args.model, payload(long_gen), max_tokens=256)
    walls_e: list[float] = []
    errors_e: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(ask_long, i) for i in range(n)]
        for future in concurrent.futures.as_completed(futures):
            try:
                _, wall = future.result()
                walls_e.append(wall)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
                errors_e.append(str(error))
    spread = f"walls {min(walls_e):.2f}s..{max(walls_e):.2f}s" if walls_e else "no walls"
    print(f"E long-gen identical burst: errors={len(errors_e)} {spread} "
          f"[{'ok' if not errors_e else 'FAIL'}]")
    passed &= not errors_e

    print("concurrency battery:", "OK" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
