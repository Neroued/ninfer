"""Live cache, boundary and mixed-question validation for an existing --jev server.

All Codacus fixtures are pinned in fixtures/decision/codacus.json. The synthetic
workflow adapts https://docs.typesafe.ai/cookbooks/parallel_questions (8 Noul,
2 Choice, 3 Score questions, batched versus separate requests). No external data,
hosted API, result cache, retries, or SDK dependency is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from decision_live import Client, FIXTURES, finite, require, validate_response, validate_systemone
from decision import ARENA_EXPECTED, typed_equal


def workflow():
    # Explicit facts make the label checks reproducible without a legal/medical corpus.
    facts = {"encrypted": True, "exports_enabled": False, "audit_enabled": True,
             "backup_enabled": True, "outage_active": False, "deletion_allowed": True,
             "approval_required": True, "public_access": False}
    state = {"service": "Orchid", "controls": facts, "owner": "operations",
             "region": "eu", "impact_level": 1, "urgency_level": 2, "risk_level": 0,
             "note": "Test record. Judge only the named fields, not typical service behavior."}
    questions = {key: {"type": "noul",
        "instructions": {"question": f"Is controls.{key} true?", "rule": "Read the boolean literally."}}
        for key in facts}
    for key, choices in (("owner", ["operations", "billing", "engineering"]),
                         ("region", ["eu", "us", "apac"])):
        questions[key] = {"type": "choice", "instructions": f"Select the exact {key} field.",
                          "criteria": {value: {key: value} for value in choices}}
    for key in ("impact_level", "urgency_level", "risk_level"):
        questions[key] = {"type": "score", "instructions": f"Select the level equal to {key}.",
                          "criteria": [{"value": i} for i in range(3)]}
    return {"state": state, "questions": questions}


def distribution(answer):
    return ({"true": answer["noul"], "false": 1 - answer["noul"]}
            if answer["type"] == "noul" else answer["probabilities"])


def answer_delta(left, right):
    require(set(left) == set(right), "answer IDs changed")
    return max((abs(p - distribution(right[key])[label])
                for key, answer in left.items()
                for label, p in distribution(answer).items()), default=0)


def decision_difference(left, right):
    require(len(left["results"]) == len(right["results"]), "context count changed")
    delta, changes = 0, []
    for index, (a, b) in enumerate(zip(left["results"], right["results"])):
        require(set(a["fields"]) == set(b["fields"]), "field names changed")
        for key, field in a["fields"].items():
            other = b["fields"][key]
            delta = max(delta, abs(field["probability"] - other["probability"]))
            if not typed_equal(field["value"], other["value"]):
                changes.append({"context": index, "field": key,
                                "left": field["value"], "right": other["value"]})
    # Decision exposes selected-value probabilities, not complete distributions.
    # Different winners must be reported alongside, not hidden by a small delta.
    return delta, changes


def workflow_labels(request, response):
    wrong = []
    for key, answer in response["answers"].items():
        kind = answer["type"]
        expected = (request["state"]["controls"][key] if kind == "noul"
                    else request["state"][key])
        actual = (answer["noul"] >= .5 if kind == "noul" else answer["choice"]
                  if kind == "choice" else int(max(answer["probabilities"],
                                                  key=answer["probabilities"].get)))
        if actual != expected:
            wrong.append({"question": key, "expected": expected, "actual": actual})
    return wrong


def validate_cache_refresh(cold, warm, refreshed):
    first, hit, fresh = (r["usage"] for r in (cold, warm, refreshed))
    require(first["cached_tokens"] > 0, "cold request did not share its branch prefixes")
    require(hit["computed_tokens"] < first["computed_tokens"], "warm request did not reuse its root")
    for key in ("computed_tokens", "cached_tokens"):
        require(fresh[key] == first[key], f"refresh changed cold {key}")
    require(refreshed["timings"]["forward_batches"] == cold["timings"]["forward_batches"],
            "refresh changed cold forward work")


class Run:
    def __init__(self, client, output):
        self.client, self.output = client, output
        self.report = {"complete": False, "records": [], "comparisons": [], "quality": [],
                       "cache_work": []}

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")

    def call(self, route, name, request, status=200):
        record = {"route": route, "case": name, "ok": False}
        start = time.perf_counter()
        response = None
        try:
            actual, response = self.client.call("/v1/" + route, request)
            record.update(status=actual, response=response)
            require(actual == status, f"expected HTTP {status}, got {actual}")
            if status == 200:
                (validate_response if route == "decision" else validate_systemone)(request, response)
            else:
                require(isinstance(response.get("detail" if route == "systemone" else "error"),
                                   list if route == "systemone" else dict), "missing validation error")
            record["ok"] = True
        except Exception as error:
            record["error"] = str(error)
        record["wall_ms"] = (time.perf_counter() - start) * 1000
        self.report["records"].append(record)
        self.save()
        print(f"{'PASS' if record['ok'] else 'FAIL'} {route} {name}", flush=True)
        return response if record["ok"] and status == 200 else None

    def compare(self, name, delta, tolerance=.02, required=True, **details):
        self.report["comparisons"].append({"case": name, "max_probability_delta": delta,
            "tolerance": tolerance if required else None, "required": required,
            "ok": (delta <= tolerance and not details.get("changed_values")) if required else None,
            **details})
        self.save()


def codacus(run):
    presets = json.loads(FIXTURES.read_text(encoding="utf-8"))["presets"]
    for preset in presets:
        for mode in ("auto", "tree", "greedy"):
            responses = []
            for cache in (False, True):
                payload = {**preset["request"], "mode": mode, "cache_prompt": cache}
                response = run.call("decision", f"{preset['id']}:{mode}:cache={cache}", payload)
                responses.append(response)
                if response and preset["id"] == "arena":
                    actual = response["results"][0]["decision"]
                    run.report["quality"].append({"case": f"arena:{mode}:cache={cache}",
                        "wrong": [key for key, value in ARENA_EXPECTED.items()
                                  if not typed_equal(actual[key], value)]})
            if all(responses):
                delta, changes = decision_difference(*responses)
                run.compare(f"cache:{preset['id']}:{mode}", delta, changed_values=changes)


def cache_reuse(run):
    # Exact request from Codacus's parallel-decision README at ad129b08,
    # excluding the example's model selector (the server owns its resident model).
    payload = {"instructions": "Answer each question about this support request from its state.",
        "schema": {
            "category": {"type": "enum", "choices": ["billing", "technical", "cancellation", "other"],
                         "description": "What type of support request is this?"},
            "urgent": {"type": "boolean", "description": "Does this need urgent handling?"},
            "priority": {"type": "enum", "choices": ["low", "medium", "high", "critical"],
                         "description": "Rate support priority."}},
        "contexts": ["I was charged twice and need this fixed today."]}
    for mode in ("auto", "tree", "greedy"):
        first = run.call("decision", f"readme:{mode}",
                         {**payload, "mode": mode, "cache_prompt": False})
        second = run.call("decision", f"readme:{mode}:repeat", {**payload, "mode": mode})
        refreshed = run.call("decision", f"readme:{mode}:refresh",
                             {**payload, "mode": mode, "cache_prompt": False})
        if first and second:
            delta, changes = decision_difference(first, second)
            run.compare(f"readme:{mode}:repeat", delta, changed_values=changes)
        if first and refreshed:
            delta, changes = decision_difference(first, refreshed)
            run.compare(f"readme:{mode}:refresh", delta, changed_values=changes)
        if first and second and refreshed:
            check = {"case": f"readme:{mode}:refresh", "ok": False}
            try:
                validate_cache_refresh(first, second, refreshed)
                check["ok"] = True
            except ValueError as error:
                check["error"] = str(error)
            run.report["cache_work"].append(check)
            run.save()
    # Interleave an unrelated rubric; then require the original request to recover.
    run.call("systemone", "cache:interleave", {"state": "A blue circle.", "questions": {
        "shape": {"type": "choice", "criteria": {"circle": None, "square": None}}}})
    recovered = run.call("decision", "readme:greedy:recovered", {**payload, "mode": "greedy"})
    if second and recovered:
        delta, changes = decision_difference(second, recovered)
        run.compare("readme:cache-invalidation", delta, changed_values=changes)


def boundaries(run):
    d = {"contexts": ["x"], "schema": {"x": {"type": "boolean", "description": "True?"}}}
    s = {"state": "x", "questions": {"x": {"type": "noul"}}}
    bad_d = {"contexts-type": {"contexts": [1]}, "contexts-257": {"contexts": [""] * 257},
        "context-bytes": {"contexts": ["x" * (2 ** 20 + 1)]}, "cache-type": {"cache_prompt": 1},
        "tree-low": {"tree_max": 0}, "tree-high": {"tree_max": 256},
        "field-count": {"schema": {f"q{i}": d["schema"]["x"] for i in range(33)}},
        "duplicate-enum": {"schema": {"x": {"type": "enum", "choices": ["a", "a"], "description": "x"}}}}
    for name, change in bad_d.items():
        run.call("decision", name, {**d, **change}, 400)
    for name, change in {"top-member": {"extra": 1}, "state-type": {"state": True},
                        "question-count": {"questions": {f"q{i}": {"type": "noul"} for i in range(33)}},
                        "state-bytes": {"state": "x" * (2 ** 20 + 1)}, "model-type": {"model": 1}}.items():
        run.call("systemone", name, {**s, **change}, 422)
    bad_questions = {"unknown-member": {"type": "noul", "extra": True},
        "instructions-type": {"type": "noul", "instructions": 4},
        "noul-criteria": {"type": "noul", "criteria": {"maybe": "x"}},
        "choice-256": {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        "empty-label": {"type": "choice", "criteria": {"": None}},
        "numeric-description": {"type": "choice", "criteria": {"a": 3}},
        "score-11": {"type": "score", "criteria": ["x"] * 11},
        "score-entry": {"type": "score", "criteria": [0, 1]}}
    for name, question in bad_questions.items():
        run.call("systemone", name, {**s, "questions": {"x": question}}, 422)
    singleton = {"type": "enum", "choices": ["only"], "description": "Only choice."}
    run.call("decision", "256-contexts-32-fields", {"contexts": [""] * 256,
        "schema": {f"q{i}": singleton for i in range(32)}})
    run.call("systemone", "32-questions", {"state": {}, "questions": {
        f"q{i}": {"type": "choice", "criteria": {"only": None}} for i in range(32)}})


def systemone_workflow(run, repeats):
    payload = workflow()
    batches, singles, batch_ms, single_ms = [], [], [], []
    for repeat in range(repeats):
        batch = run.call("systemone", f"workflow:batch:{repeat}", payload)
        batch_ms.append(run.report["records"][-1]["wall_ms"])
        answers, elapsed = {}, 0
        for key, question in payload["questions"].items():
            response = run.call("systemone", f"workflow:single:{repeat}:{key}",
                                {**payload, "questions": {key: question}})
            elapsed += run.report["records"][-1]["wall_ms"]
            if response:
                answers.update(response["answers"])
        single_ms.append(elapsed)
        if batch:
            batches.append(batch["answers"])
            run.report["quality"].append({"case": f"workflow:{repeat}",
                                         "wrong": workflow_labels(payload, batch)})
        if len(answers) == len(payload["questions"]):
            singles.append(answers)
            run.report["quality"].append({"case": f"workflow:single:{repeat}",
                "wrong": workflow_labels(payload, {"answers": answers})})
            if batch:
                # Single and multi-question prompts intentionally use different ordering.
                # Measure disagreement instead of pretending this is numerical identity.
                run.compare(f"workflow:batch-vs-single:{repeat}",
                            answer_delta(batch["answers"], answers), required=False)
    for name, responses in (("batch", batches), ("single", singles)):
        for index, response in enumerate(responses[1:], 1):
            run.compare(f"workflow:repeat:{name}:{index}", answer_delta(responses[0], response))
    if batches:
        renamed = {f"private_{i}": question for i, question in enumerate(payload["questions"].values())}
        response = run.call("systemone", "workflow:renamed-ids", {**payload, "questions": renamed})
        if response:
            remapped = {key: response["answers"][f"private_{i}"]
                        for i, key in enumerate(payload["questions"])}
            run.compare("workflow:renamed-ids", answer_delta(batches[0], remapped))
    run.report["workflow_timing"] = {"batch_median_ms": statistics.median(batch_ms),
                                    "singles_total_median_ms": statistics.median(single_ms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--section", choices=("all", "codacus", "cache", "boundaries", "workflow"), default="all")
    args = parser.parse_args()
    if args.repeats < 1 or not finite(args.timeout) or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    run = Run(Client(args.base_url, args.timeout), args.output)
    run.report.update(base_url=args.base_url, section=args.section, repeats=args.repeats)
    run.save()
    try:
        require(run.client.call("/health")[0] == 200, "unhealthy before validation")
        if args.section in ("all", "codacus"):
            codacus(run)
        if args.section in ("all", "cache"):
            cache_reuse(run)
        if args.section in ("all", "boundaries"):
            boundaries(run)
        if args.section in ("all", "workflow"):
            systemone_workflow(run, args.repeats)
        require(run.client.call("/health")[0] == 200, "unhealthy after validation")
        run.report["complete"] = True
    finally:
        run.save()
    failed = (any(not r["ok"] for r in run.report["records"]) or
              any(not r["ok"] for r in run.report["cache_work"]) or
              any(r["required"] and not r["ok"] for r in run.report["comparisons"]) or
              any(r["wrong"] for r in run.report["quality"]))
    print(json.dumps({"requests": len(run.report["records"]), "failed": failed}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
