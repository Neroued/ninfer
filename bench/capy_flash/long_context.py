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
long_context.py concurrent URL MODEL OUT.jsonl --text FILE --tokenizer tokenizer.json --sizes 32000 --lanes 1,2,4
    [--max-tokens 512] [--server-log serve.log] [--stagger S] [--warm]
    For each lane count L and size: L fresh conversations (distinct slices of FILE, the decode request) sent at
    once on L connections, streamed, temperature 0. Per request: time to the first token, decode rate (tokens
    after the first over the time from the first to the last token: it includes the rounds another lane's
    prompt took), content and reasoning; per group: the aggregate rate (all completion tokens over the time from
    the sends to the last token), the steady rate (tokens of every lane streamed while all of them were
    decoding, from the last first token to the first last token, over that span; streamed pieces scaled to the
    completion tokens), MemAvailable before and after, and with --server-log the last
    `active KV in RAM:` line (the Host KV planes' RAM of a --kv-ram-active server). --stagger S sends lane i
    S x i seconds after lane 0, so the server admits the lanes in order (each request records its send time and
    the arrival time of every streamed piece, `sent_s` and `pieces_s`, from the group's start). --warm first sends each lane's request
    alone with max_tokens 1 (its prompt then stays in the context cache), so the measured group resumes cached
    conversations and every lane decodes from the start (the steady span is the whole decode).
long_context.py summarize REQUEST_LOG [OUT.jsonl]
    Server-side decode rate per request (completion tokens / decode seconds from `--request-log-jsonl`),
    prompt tokens, accepted MTP drafts per round and, with OUT.jsonl, the expert slots and KV bytes that
    `decode` recorded for the request with the same client label.
"""

import argparse
import json
import os
import re
import threading
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


def last_line(path, prefix):
    if not path:
        return None
    try:
        with open(path, errors="replace") as f:
            lines = [line.strip() for line in f if line.startswith(prefix)]
    except OSError:
        return None
    return lines[-1] if lines else None


def last_elastic(path):
    return last_line(path, "elastic KV (")


def mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


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
            t0, first, usage, error = time.perf_counter(), None, None, None
            done, finished = False, False
            try:
                with post(args.url, body, client=client) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                            break
                        data = json.loads(payload)
                        if data.get("error"):
                            error = "stream error: " + json.dumps(data["error"])[:280]
                            break
                        usage = data.get("usage") or usage
                        for choice in data.get("choices", []):
                            finished = finished or choice.get("finish_reason") is not None
                            delta = choice.get("delta", {})
                            if first is None and (
                                delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                            ):
                                first = time.perf_counter() - t0
                # No victory snacks until the capybara crosses the stream.
                if error is None and not (done and finished):
                    error = "incomplete stream: missing finish_reason or [DONE]"
            except Exception as e:  # noqa: BLE001 - recorded, the next size still runs
                error = repr(e)[:300]
            finally:
                elapsed = time.perf_counter() - t0  # Capybara nap time is not response time.
            time.sleep(1.0)  # let the server print its post-request elastic line
            record = {
                "n": n,
                "client": client,
                "usage": usage,
                "wall_s": elapsed,
                "ttft_s": first,
                "elastic": last_elastic(args.server_log),
                "error": error,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps(record), flush=True)


def stream(url, body, client, record, delay=0.0):
    """Sends one streamed request; fills record with send/first/last times, usage, content."""
    if delay > 0:
        time.sleep(delay)
    record["sent"] = time.time()
    content, reasoning = [], []
    try:
        with post(url, body, client=client) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                data = json.loads(line[5:])
                if data.get("error"):
                    record["error"] = json.dumps(data["error"])[:300]
                record["usage"] = data.get("usage") or record.get("usage")
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece or thought:
                        now = time.time()
                        record.setdefault("first", now)
                        record["last"] = now
                        record.setdefault("pieces", []).append(now)
                    content.append(piece)
                    reasoning.append(thought)
    except Exception as e:  # noqa: BLE001 - recorded, the other lanes still run
        record["error"] = repr(e)[:300]
    record["content"] = "".join(content)
    record["reasoning"] = "".join(reasoning)


def concurrent(args, tokenizer, text):
    run = f"{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
    start = 0
    with open(args.out, "a") as out:
        for lanes in args.lanes:
            for n in args.sizes:
                bodies, clients = [], []
                for lane in range(lanes):
                    prompt, start = cut(tokenizer, text, start, n)
                    clients.append(f"long_context/{run}/c{lanes}/{n}/{lane}")
                    bodies.append(
                        {
                            "model": args.model,
                            "messages": [{"role": "user", "content": prompt + ASK}],
                            "max_tokens": args.max_tokens,
                            "temperature": 0,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        }
                    )
                if args.warm:
                    for i in range(lanes):
                        warm = dict(bodies[i], max_tokens=1)
                        stream(args.url, warm, clients[i] + "/warm", {})
                records = [{} for _ in range(lanes)]
                memory_before = mem_available_gib()
                threads = [
                    threading.Thread(
                        target=stream, args=(args.url, bodies[i], clients[i], records[i], args.stagger * i)
                    )
                    for i in range(lanes)
                ]
                t0 = time.time()
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                memory_after = mem_available_gib()
                time.sleep(1.0)  # let the server print its post-request lines
                tokens = [((r.get("usage") or {}).get("completion_tokens") or 0) for r in records]
                lasts = [r["last"] for r in records if "last" in r]
                firsts = [r["first"] for r in records if "first" in r]
                # The span when every lane was decoding: from the last first token to the first end.
                both = (min(lasts) - max(firsts)) if len(lasts) == lanes and len(firsts) == lanes else None
                in_span = 0.0
                if both and both > 0:
                    lo, hi = max(firsts), min(lasts)
                    for r, k in zip(records, tokens):
                        pieces = r.get("pieces", [])
                        inside = sum(1 for t in pieces if lo <= t <= hi)
                        in_span += inside * (k / len(pieces)) if pieces else 0.0
                group = {
                    "lanes": lanes,
                    "n": n,
                    "requests": [
                        {
                            "client": clients[i],
                            "usage": records[i].get("usage"),
                            "ttft_s": records[i]["first"] - t0 if "first" in records[i] else None,
                            "decode_tok_s": (tokens[i] - 1) / (records[i]["last"] - records[i]["first"])
                            if "first" in records[i] and records[i]["last"] > records[i]["first"]
                            else None,
                            "error": records[i].get("error"),
                            "content": records[i]["content"],
                            "reasoning": records[i]["reasoning"],
                            "sent_s": records[i].get("sent", t0) - t0,
                            "pieces_s": [round(t - t0, 4) for t in records[i].get("pieces", [])],
                        }
                        for i in range(lanes)
                    ],
                    "aggregate_tok_s": sum(tokens) / (max(lasts) - t0) if lasts else None,
                    "steady_tok_s": in_span / both if both and both > 0 else None,
                    "steady_s": both,
                    "wall_s": (max(lasts) - t0) if lasts else None,
                    "mem_available_gib": [memory_before, memory_after],
                    "host_kv": last_line(args.server_log, "active KV in RAM:"),
                }
                out.write(json.dumps(group, ensure_ascii=False) + "\n")
                out.flush()
                brief = {k: v for k, v in group.items() if k != "requests"}
                brief["ttft_s"] = [r["ttft_s"] for r in group["requests"]]
                brief["decode_tok_s"] = [r["decode_tok_s"] for r in group["requests"]]
                brief["tokens"] = tokens
                print(json.dumps(brief), flush=True)


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
    for name in ("decode", "greedy", "concurrent"):
        p = sub.add_parser(name)
        p.add_argument("url")
        p.add_argument("model")
        p.add_argument("out")
        p.add_argument("--text", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--sizes", required=True, type=lambda s: [int(x) for x in s.split(",")])
        if name in ("decode", "concurrent"):
            p.add_argument("--max-tokens", type=int, default=512)
            p.add_argument("--server-log")
        if name == "concurrent":
            p.add_argument("--lanes", required=True, type=lambda s: [int(x) for x in s.split(",")])
            p.add_argument("--stagger", type=float, default=0.0)
            p.add_argument("--warm", action="store_true")
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
    {"decode": decode, "greedy": greedy, "concurrent": concurrent}[args.command](args, tokenizer, text)


if __name__ == "__main__":
    main()
