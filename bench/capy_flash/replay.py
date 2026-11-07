#!/usr/bin/env python3
"""Agentic replay client (T71): for each session and turn t, POST /v1/chat/completions with the system prompt, the task and
turns 0..t (fixed assistant message + tool output as a user message), stream, greedy, max_tokens MAXNEW.
Measures client-side TTFT (first streamed token) and decode tokens/s (usage.completion_tokens - 1 over first->last).
Usage: replay.py URL MODEL sessions.json out.jsonl [maxnew]"""

import json
import sys
import time
import urllib.request

url, model, sess_path, out_path = sys.argv[1:5]
maxnew = int(sys.argv[5]) if len(sys.argv) > 5 else 256
S = json.load(open(sess_path))
out = open(out_path, "a")
for si, s in enumerate(S["sessions"]):
    msgs = [{"role": "system", "content": S["system"]}, {"role": "user", "content": s["task"]}]
    for ti, t in enumerate(s["turns"]):
        msgs = msgs + [{"role": "assistant", "content": t["assistant"]}, {"role": "user", "content": t["tool"]}]
        body = json.dumps(
            {
                "model": model,
                "messages": msgs,
                "max_tokens": maxnew,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        t0 = time.time()
        first = last = None
        usage = None
        n_chunks = 0
        err = None
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("error"):
                        # An error after the stream started (#20): the request failed, not a short reply.
                        err = "stream error: " + json.dumps(d["error"])[:280]
                        break
                    if d.get("usage"):
                        usage = d["usage"]
                    for c in d.get("choices", []):
                        delta = c.get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                            now = time.time()
                            first = first or now
                            last = now
                            n_chunks += 1
        except Exception as e:
            err = repr(e)[:300]
        t1 = time.time()
        ct = (usage or {}).get("completion_tokens", n_chunks)
        pt = (usage or {}).get("prompt_tokens")
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        rec = {
            "session": si,
            "turn": ti,
            "prompt_tokens": pt,
            "cached_tokens": cached,
            "completion_tokens": ct,
            "ttft_s": (first - t0) if first else None,
            "total_s": t1 - t0,
            "decode_tps": (ct - 1) / (last - first) if first and last and last > first and ct > 1 else None,
            "error": err,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(
            f"s{si} t{ti:02d} prompt {pt} cached {cached} ttft {rec['ttft_s'] and round(rec['ttft_s'], 2)} s  decode {rec['decode_tps'] and round(rec['decode_tps'], 1)} t/s  total {round(t1 - t0, 1)} s {err or ''}",
            flush=True,
        )
