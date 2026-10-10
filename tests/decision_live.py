"""Live Decision/SystemOne contracts and finite, lossless chat replay (stdlib only).

No Twitch connection, remote judge, model loading, automatic retries or answer cache.
Every event is independently scored. Run against an explicitly selected --jev server.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/smoke"))
from decision import FIXTURES, finite, require, validate_response

CHAT = Path(__file__).parent / "fixtures/decision/chat.json"


def probability(value):
    return finite(value) and 0 <= value <= 1


def validate_systemone(request, response):
    require(isinstance(response.get("model"), str) and response["model"], "missing model")
    require(set(response["answers"]) == set(request["questions"]), "wrong answer IDs")
    for name, question in request["questions"].items():
        answer = response["answers"][name]
        kind = question["type"]
        require(answer["type"] == kind, "wrong answer type")
        if kind == "noul":
            require(set(answer) == {"type", "noul"} and probability(answer["noul"]), "bad noul")
            continue
        criteria = question["criteria"]
        keys = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
        probs = answer["probabilities"]
        require(set(probs) == keys and all(probability(p) for p in probs.values()),
                "invalid distribution")
        require(abs(sum(probs.values()) - 1) < 1e-8, "distribution not normalized")
        require(probability(answer["confidence"]), "invalid confidence")
        if kind == "choice":
            require(answer["choice"] in keys and
                    abs(probs[answer["choice"]] - max(probs.values())) < 1e-8, "not argmax")
            n = len(probs)
            expected = 1 if n == 1 else (max(probs.values()) * n - 1) / (n - 1)
        else:
            require(answer["legend"] == {str(i): c for i, c in enumerate(criteria)}, "bad legend")
            mean = sum(int(k) * p for k, p in probs.items())
            require(finite(answer["score"]) and abs(answer["score"] - mean) < 1e-8, "bad mean")
            mode = max(range(len(criteria)), key=lambda i: probs[str(i)])
            spread = sum(p * abs(int(k) - mode) for k, p in probs.items())
            uniform = sum(abs(i - (len(criteria) - 1) / 2) for i in range(len(criteria))) / len(criteria)
            expected = max(0, min(1, 1 - spread / uniform))
        require(abs(answer["confidence"] - expected) < 1e-8, "bad confidence formula")
    usage = response["usage"]
    require(type(usage["input_tokens"]) is int and usage["input_tokens"] >= 0 and
            type(usage["output_tokens"]) is int and usage["output_tokens"] == 0, "bad usage")


class Client:
    def __init__(self, base_url, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Local endpoints must not be routed through environment-configured proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, path, payload=None):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key := os.environ.get("NINFER_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.base_url + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers=headers)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)


def systemone_cases():
    choice = {"type": "choice", "instructions": "Choose the applicable team.",
              "criteria": {"billing": "Payments", "technical": {"area": "Software"}, "other": None}}
    noul = {"type": "noul", "instructions": "Is this urgent?",
            "criteria": {"true": "Time critical", "false": "No rush"}}
    score = {"type": "score", "instructions": "How severe?", "criteria": ["Low", "Medium", "High"]}
    yield "mixed", {"state": {"message": "Charged twice", "amount": 42},
                    "questions": {"route": choice, "urgent": noul, "severity": score}}
    for name, question in (("choice", choice), ("noul", noul), ("score", score)):
        yield name, {"state": "Charged twice. No rush.", "questions": {name: question}}
    yield "null", {"state": None, "questions": {"nullable": {"type": "noul"}}}
    yield "array", {"state": ["event", {"id": 2}], "questions": {"properties": {
        "type": "choice", "instructions": ["Pick", {"only": "valid option"}],
        "criteria": {"quoted\"label\\path": None}}}}
    yield "structured-score", {"state": "Neutral", "questions": {"score": {
        "type": "score", "instructions": None, "criteria": [{"level": "low"}, None]}}}
    yield "ten-levels", {"state": "A minor inconvenience", "questions": {"score": {
        "type": "score", "criteria": [f"Severity {i}" for i in range(10)]}}}


def run_contracts(client, routes):
    records = []

    def check(route, name, payload, expected=200):
        started = time.perf_counter()
        record = {"route": route, "case": name, "ok": False}
        try:
            status, response = client.call("/v1/" + route, payload)
            record["status"] = status
            require(status == expected, f"expected HTTP {expected}, got {status}: {response}")
            if status == 200:
                (validate_response if route == "decision" else validate_systemone)(payload, response)
            elif route == "systemone":
                require(isinstance(response.get("detail"), list), "missing validation detail")
            else:
                require(isinstance(response.get("error"), dict), "missing error object")
            record["ok"] = True
        except Exception as error:
            record["error"] = str(error)
        record["wall_ms"] = (time.perf_counter() - started) * 1000
        records.append(record)
        print(json.dumps(record), flush=True)

    if "decision" in routes:
        presets = json.loads(FIXTURES.read_text(encoding="utf-8"))["presets"]
        for preset in presets:
            for mode in ("auto", "tree", "greedy"):
                check("decision", preset["id"] + ":" + mode, {**preset["request"], "mode": mode})
        small = next(p["request"] for p in presets if p["id"] == "routing")
        check("decision", "cache-off", {**small, "cache_prompt": False})
        for name, change in (("empty-contexts", {"contexts": []}),
                             ("empty-schema", {"schema": {}}), ("invalid-mode", {"mode": "wrong"})):
            check("decision", name, {**small, **change}, 400)
    if "systemone" in routes:
        for name, payload in systemone_cases():
            check("systemone", name, payload)
        invalid = [({}, "missing-state"), ({"state": 42, "questions": {"x": {"type": "noul"}}}, "scalar"),
                   ({"state": "x", "questions": {}}, "empty-questions"),
                   ({"state": "x", "questions": {"x": {"type": "score", "criteria": ["one"]}}}, "score-bound"),
                   ({"state": "x", "questions": {"x": {"type": "choice", "criteria": {}}}}, "empty-choices"),
                   ({"state": "x", "questions": {"x": {"type": "unknown"}}}, "unknown-type")]
        for payload, name in invalid:
            check("systemone", name, payload, 422)
    return records


def chat_request(route, fixture, text):
    if route == "decision":
        return {"instructions": fixture["policy"], "contexts": [json.dumps({"message": text}, ensure_ascii=False)],
                "cache_prompt": True, "mode": "tree", "schema": {"category": {"type": "enum",
                "choices": list(fixture["criteria"]), "description": "Primary intent of the target message."}}}
    # Policy belongs in the reusable rubric, never in each changing state.
    return {"state": {"message": text}, "questions": {"category": {"type": "choice",
            "instructions": fixture["policy"], "criteria": fixture["criteria"]}}}


def classify(client, route, fixture, message):
    payload = chat_request(route, fixture, message["text"])
    status, response = client.call("/v1/" + route, payload)
    require(status == 200, f"HTTP {status}: {response}")
    (validate_response if route == "decision" else validate_systemone)(payload, response)
    category = (response["results"][0]["decision"]["category"] if route == "decision"
                else response["answers"]["category"]["choice"])
    return {"category": category, "expected": message["expected"],
            "usage": response["usage"], "timings": response.get("timings")}


async def replay(events, rate, concurrency, invoke):
    """Finite open-loop arrivals; unlimited pending queue, bounded in-flight work, full drain.

    invoke is async and returns a result dict or raises. Errors get explicit event records,
    not retries or dropped events. rate=0 is a saturation run, not a real-time rate claim.
    """
    queue = asyncio.Queue()
    origin = time.perf_counter()
    results, peak_pending = [], 0

    async def worker():
        while (item := await queue.get()) is not None:
            index, event, due = item
            started = time.perf_counter()
            record = {"event": index, "ok": False, "queue_ms": (started - due) * 1000}
            try:
                record.update(await invoke(event))
                record["ok"] = True
            except Exception as error:
                record["error"] = str(error)
            finally:
                ended = time.perf_counter()
                record.update(request_ms=(ended - started) * 1000, e2e_ms=(ended - due) * 1000)
                results.append(record)
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        for index, event in enumerate(events):
            due = origin + index / rate if rate else origin
            # Some event loops can wake early by their clock resolution. Do not
            # enqueue before the scheduled arrival or report negative queue age.
            while (delay := due - time.perf_counter()) > 0:
                await asyncio.sleep(delay)
            queue.put_nowait((index, event, due))
            peak_pending = max(peak_pending, queue.qsize())
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    require(sorted(r["event"] for r in results) == list(range(len(events))), "lost/duplicated events")
    elapsed = time.perf_counter() - origin
    return {"rate": rate, "concurrency": concurrency, "received": len(events),
            "completed": sum(r["ok"] for r in results), "errors": sum(not r["ok"] for r in results),
            "dropped": 0, "peak_pending": peak_pending, "elapsed_s": elapsed,
            "completed_per_s": sum(r["ok"] for r in results) / elapsed,
            "records": sorted(results, key=lambda r: r["event"])}


async def replay_http(client, route, fixture, events, rate, concurrency):
    # asyncio.to_thread's default pool can silently cap a requested stress run
    # below --concurrency (for example, twelve workers on an eight-core client).
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        async def invoke(event):
            return await loop.run_in_executor(pool, classify, client, route, fixture, event)
        return await replay(events, rate, concurrency, invoke)


def summarize(stage):
    summary = {k: v for k, v in stage.items() if k != "records"}
    good = [r for r in stage["records"] if r["ok"]]
    for key in ("request_ms", "queue_ms", "e2e_ms"):
        values = sorted(r[key] for r in good)
        for p in (50, 95, 99):
            summary[f"{key}_p{p}"] = values[math.ceil(len(values) * p / 100) - 1] if values else None
    summary["label_matches"] = sum(r.get("category") == r.get("expected") for r in good)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--routes", nargs="+", choices=("decision", "systemone"), default=["decision", "systemone"])
    parser.add_argument("--section", choices=("all", "contracts", "replay"), default="all")
    parser.add_argument("--rates", nargs="+", type=float, default=[8, 24, 48])
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60, help="per HTTP request, not queued event age")
    parser.add_argument("--output", type=Path, default=Path(".local/decision-live.json"))
    parser.add_argument("--build-label", default="unspecified", help="identify the actual served revision")
    parser.add_argument("--min-throughput-ratio", type=float,
                        help="optional SystemOne/Decision completion-rate gate on saturation runs")
    args = parser.parse_args()
    if (args.count < 1 or args.concurrency < 1 or not finite(args.timeout) or args.timeout <= 0 or
            any(not finite(r) or r < 0 for r in args.rates)):
        parser.error("count/concurrency/timeout must be positive; rates finite and nonnegative")
    if args.min_throughput_ratio is not None and (not finite(args.min_throughput_ratio) or
            args.min_throughput_ratio <= 0 or set(args.routes) != {"decision", "systemone"} or
            args.section == "contracts" or 0 not in args.rates):
        parser.error("throughput gate requires positive ratio, both routes and a rate=0 replay")
    client = Client(args.base_url, args.timeout)
    report = {"build": args.build_label, "started_unix": time.time(), "endpoint": args.base_url,
              "contracts": [], "stages": [], "complete": False,
              "policy": "independent messages; no dedup, answer cache, retries or queue drops"}
    failed = False

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        status, report["health_before"] = client.call("/health")
        require(status == 200, "server not healthy")
        if args.section != "replay":
            report["contracts"] = run_contracts(client, args.routes)
            failed |= any(not r["ok"] for r in report["contracts"])
            save()
        if args.section != "contracts":
            fixture = json.loads(CHAT.read_text(encoding="utf-8"))
            events = [copy.deepcopy(fixture["messages"][i % len(fixture["messages"])]) for i in range(args.count)]
            for route in args.routes:
                for message in events[:2]:
                    classify(client, route, fixture, message)  # excluded warmups

                for rate, concurrency, batch, name in [(0, 1, events[:16], "serial")] + [
                        (r, args.concurrency, events, "stream" if r else "saturation") for r in args.rates]:
                    stage = asyncio.run(replay_http(client, route, fixture, batch, rate, concurrency))
                    stage.update(route=route, name=name)
                    report["stages"].append(stage)
                    failed |= stage["errors"] != 0
                    print(json.dumps(summarize(stage)), flush=True)
                    save()
            if args.min_throughput_ratio is not None:
                speed = {s["route"]: s["completed_per_s"] for s in report["stages"] if s["name"] == "saturation"}
                ratio = speed["systemone"] / speed["decision"]
                report["throughput_ratio"] = ratio
                failed |= ratio < args.min_throughput_ratio
        status, report["health_after"] = client.call("/health")
        require(status == 200, "server not healthy after run")
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        failed = True
    finally:
        report["passed"] = not failed
        report["finished_unix"] = time.time()
        save()
    print(f"{'FAIL' if failed else 'PASS'}; report: {args.output}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
