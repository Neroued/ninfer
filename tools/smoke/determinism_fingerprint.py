"""Determinism fingerprint for a running ninfer-serve.

Two checks, both greedy (temperature 0, thinking off):

  solo   — the same long-generation prompt twice on an idle server; the two continuations must
           be byte-identical (a difference means a real nondeterminism source: a race or an
           unguarded reduction, not batch-composition effects);
  co-load — the same prompt while N unrelated clients generate concurrently; reports the first
           token index at which the co-loaded continuation diverges from the solo run. Divergence
           here is the documented batch-composition effect (see docs/serving.md, Determinism)
           and is reported, not failed.

Usage: python3 determinism_fingerprint.py --base-url http://HOST:PORT --model <id>
       [--filler 1700] [--max-tokens 256] [--co-clients 4]
Exit 0 when the solo check holds; 1 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request


def ask(base_url: str, model: str, content: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "enable_thinking": False,
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def first_divergence(a: str, b: str) -> int:
    words_a, words_b = a.split(), b.split()
    for index, (x, y) in enumerate(zip(words_a, words_b)):
        if x != y:
            return index
    return -1 if len(words_a) == len(words_b) else min(len(words_a), len(words_b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--filler", type=int, default=1700)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--co-clients", type=int, default=4)
    args = parser.parse_args()

    salt = int(time.time())
    filler = "Audit line item with routine status and no anomalies recorded. "
    prompt = f"Fingerprint {salt}. " + filler * args.filler + " Write a short story about an audit."

    first = ask(args.base_url, args.model, prompt, args.max_tokens)
    second = ask(args.base_url, args.model, prompt, args.max_tokens)
    solo_ok = first == second
    print(f"solo: {'identical' if solo_ok else 'DIVERGED at word ' + str(first_divergence(first, second))}")

    noise = [f"Noise {salt} {i}. " + filler * (args.filler // 2) + " Write a poem about audits."
             for i in range(args.co_clients)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.co_clients + 1) as pool:
        noise_futures = [pool.submit(ask, args.base_url, args.model, n, args.max_tokens) for n in noise]
        loaded = pool.submit(ask, args.base_url, args.model, prompt, args.max_tokens).result()
        for future in noise_futures:
            future.result()
    divergence = first_divergence(first, loaded)
    print("co-load: " + ("identical to solo" if divergence == -1
                         else f"diverges from solo at word {divergence} (batch-composition effect)"))
    print("determinism fingerprint:", "OK" if solo_ok else "FAIL")
    return 0 if solo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
