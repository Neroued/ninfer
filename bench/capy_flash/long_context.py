#!/usr/bin/env python3
"""Long-context decode and greedy-identity probes for an OpenAI-compatible server (capy-flash).

long_context.py decode URL MODEL OUT.jsonl --text FILE --tokenizer tokenizer.json --sizes 16000,64000
    [--max-tokens 512] [--server-log serve.log]
    One fresh conversation per size: a slice of FILE cut to SIZE tokens (a distinct slice per size, so no
    prefix is reused), then a fixed summarization request; streamed, temperature 0. Records the usage, the
    client label the request carried (X-Ninfer-Client, unique per run and size) and, with --server-log, the
    last `elastic KV (...)` line the server printed after the request (expert slots per layer per GPU and KV
    bytes per GPU of a `--elastic` Flash-Next). Appends to OUT.jsonl.
long_context.py greedy URL MODEL OUT.jsonl --text FILE --tokenizer tokenizer.json --sizes 1000,30000
    Two greedy requests per size (thinking off with a short answer, thinking on with a longer one) over
    prefixes of FILE; records content, reasoning and completion tokens. Two runs are identical when
    their OUT files are equal (compare with `cmp`).
long_context.py summarize REQUEST_LOG [OUT.jsonl]
    Server-side decode rate per request (completion tokens / decode seconds from `--request-log-jsonl`),
    prompt tokens, accepted MTP drafts per round and, with OUT.jsonl, the expert slots and KV bytes that
    `decode` recorded for the request with the same client label.
"""

import argparse
import json
import os
import re
import time
import urllib.request

ASK = (
    "\n\nThe text above is an excerpt of Wikipedia articles. Write a long, detailed, well organised summary of "
    "it, article by article, with at least 800 words."
)
GREEDY_ASKS = [
    ("sum", "\n\nSummarize the text above in three sentences.", False, 160),
    ("q", "\n\nWhich people are mentioned in the first and in the last article above? Think it through.", True, 384),
]


def cut(tokenizer, text, start, tokens):
    """The prefix of text[start:] holding `tokens` tokens, and the end offset in `text`."""
    chunk = text[start : start + tokens * 6]
    offsets = tokenizer.encode(chunk).offsets
    end = offsets[min(tokens, len(offsets)) - 1][1]
    return chunk[:end], start + end


def last_elastic(path):
    if not path:
        return None
    try:
        with open(path, errors="replace") as f:
            lines = [line.strip() for line in f if line.startswith("elastic KV (")]
    except OSError:
        return None
    return lines[-1] if lines else None


def post(url, body, timeout=3600, client=None):
    headers = {"Content-Type": "application/json"}
    if client:
        headers["X-Ninfer-Client"] = client
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    return urllib.request.urlopen(request, timeout=timeout)


def decode(args, tokenizer, text):
    # The request log records this label with each request: summarize pairs a request with its probe
    # by it, whatever else the log or an appended OUT file holds.
    run = f"{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
    start = 0
    with open(args.out, "a") as out:
        for n in args.sizes:
            prompt, start = cut(tokenizer, text, start, n)
            client = f"long_context/{run}/{n}"
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt + ASK}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            t0, first, usage, error = time.time(), None, None, None
            try:
                with post(args.url, body, client=client) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        data = json.loads(line[5:])
                        usage = data.get("usage") or usage
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            if first is None and (
                                delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                            ):
                                first = time.time()
            except Exception as e:  # noqa: BLE001 - recorded, the next size still runs
                error = repr(e)[:300]
            time.sleep(1.0)  # let the server print its post-request elastic line
            record = {
                "n": n,
                "client": client,
                "usage": usage,
                "wall_s": time.time() - t0,
                "ttft_s": first - t0 if first else None,
                "elastic": last_elastic(args.server_log),
                "error": error,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps(record), flush=True)


def greedy(args, tokenizer, text):
    with open(args.out, "w") as out:
        for n in args.sizes:
            prompt, _ = cut(tokenizer, text, 0, n)
            for tag, ask, think, max_tokens in GREEDY_ASKS:
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt + ask}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                if not think:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                try:
                    with post(args.url, body, timeout=1800) as response:
                        reply = json.load(response)
                    message = reply["choices"][0]["message"]
                    record = {
                        "tag": f"{tag}{n}",
                        "prompt_tokens": reply["usage"]["prompt_tokens"],
                        "content": message.get("content"),
                        "reasoning": message.get("reasoning_content") or message.get("reasoning"),
                        "tokens": reply["usage"]["completion_tokens"],
                    }
                except Exception as e:  # noqa: BLE001 - recorded as a difference
                    record = {"tag": f"{tag}{n}", "error": repr(e)[:300]}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()


def summarize(args):
    with open(args.request_log) as f:
        done = [json.loads(line) for line in f if '"request_done"' in line]
    probes = {}
    if args.out:
        with open(args.out) as f:
            probes = {probe["client"]: probe for probe in map(json.loads, f) if probe.get("client")}
    for record in done:
        result, timings = record["result"], record["timings_seconds"]
        rate = result["completion_tokens"] / timings["decode"] if timings["decode"] else 0.0
        speculative = record.get("speculative") or {}
        per_round = (speculative.get("accepted_tokens") or 0) / max(1, speculative.get("rounds") or 1)
        line = f"prompt {result['prompt_tokens']:7d}: decode {rate:6.1f} tok/s ({result['completion_tokens']} tokens, accepted/round {per_round:.2f})"
        probe = probes.get(record["request"].get("client"))
        elastic = probe.get("elastic") if probe else None
        if elastic:
            slots = re.search(r"\((\d+) slots/layer/GPU\)", elastic)
            kv = re.search(r"MTP KV \d+ pages, (\d+) KV bytes", elastic)
            line += f", slots/layer/GPU {slots.group(1) if slots else '?'}, KV {int(kv.group(1)) / 2**20 if kv else -1:.0f} MiB/GPU"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("decode", "greedy"):
        p = sub.add_parser(name)
        p.add_argument("url")
        p.add_argument("model")
        p.add_argument("out")
        p.add_argument("--text", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--sizes", required=True, type=lambda s: [int(x) for x in s.split(",")])
        if name == "decode":
            p.add_argument("--max-tokens", type=int, default=512)
            p.add_argument("--server-log")
    p = sub.add_parser("summarize")
    p.add_argument("request_log")
    p.add_argument("out", nargs="?")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args)
        return
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    (decode if args.command == "decode" else greedy)(args, tokenizer, text)


if __name__ == "__main__":
    main()
