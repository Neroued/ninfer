#!/usr/bin/env python3
"""Paired diagnostic quality fixtures. Passing these is not KVMem equivalence.

Python 3.11, standard library only. Freeze a fixture file once, then run that exact
file against each engine configuration. Reports preserve complete requests/responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import time
import traceback
import urllib.error
import urllib.request

from kvmem_suite import Suite, validate_json
from kvmem_quality_stats import paired_accuracy, paired_cluster_accuracy


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def fixtures(rows=400, seed=74191):
    rng = random.Random(seed)
    cases = []
    instruction = ('Use only the supplied records. Reply with exactly one JSON object '
                   'with one string field "answer". No prose or markdown. If the requested '
                   'value is absent, use "NOT_FOUND". Later updates supersede earlier values.')
    for family in ("needle", "tool_tail", "two_hop", "updated", "absent"):
        for depth in (0.1, 0.5, 0.9):
            target = f"project-{rng.randrange(100000, 999999)}"
            gold = f"KEY-{rng.randrange(10000000, 99999999)}"
            records = [f"Record {i}: project-{rng.randrange(100000, 999999)} has access key "
                       f"KEY-{rng.randrange(10000000, 99999999)}. Inspection is complete."
                       for i in range(rows)]
            position = int(rows * depth)
            question = f"What is the current access key for {target}?"
            if family == "two_hop":
                team = f"team-{rng.randrange(10000, 99999)}"
                records[position] = f"Record {position}: {target} is owned by {team}."
                other = (position + rows // 2) % rows
                records[other] = f"Record {other}: {team} uses access key {gold}."
                question = f"What access key is used by the team that owns {target}?"
            elif family == "absent":
                question = f"What is the emergency satellite address for {target}?"
                gold = "NOT_FOUND"
            else:
                records[position] = f"Record {position}: {target} has access key {gold}."
                if family == "updated":
                    old = f"KEY-{rng.randrange(10000000, 99999999)}"
                    records[position] = f"Record {position}: {target} has access key {old}."
                    new_position = min(rows - 1, position + max(1, rows // 12))
                    records[new_position] = f"Record {new_position}: Update: {target} now has access key {gold}; the earlier key is revoked."
            messages = [{"role": "system", "content": instruction},
                        {"role": "user", "content": "Historical records:\n" + "\n".join(records)},
                        {"role": "assistant", "content": "I have received the records."},
                        {"role": "user", "content": question}]
            if family == "tool_tail":
                messages.extend([
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "diagnostic", "type": "function", "function": {
                            "name": "read_diagnostics", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "diagnostic", "content":
                        "Unrelated equipment diagnostics; no access keys in this output.\n" +
                        "\n".join(f"Sensor {i}: nominal; coolant pressure stable; latency 18 ms."
                                  for i in range(180))}])
            cases.append({"id": f"{family}-{int(depth * 100):02d}", "family": family,
                          "depth": depth, "gold": gold, "messages": messages})
    return {"schema": 1, "kind": "diagnostic-only", "seed": seed, "rows": rows,
            "cases": cases}


def grade(response, gold):
    choices = response.get("choices", [])
    if "error" in response or not choices:
        return {"correct": False, "reason": "missing completion/error"}
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content")
    if choice.get("finish_reason") != "stop" or not isinstance(content, str):
        return {"correct": False, "reason": "incomplete or non-text answer"}
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result
    try:
        parsed = json.loads(content, object_pairs_hook=unique_fields)
    except (ValueError, TypeError):
        return {"correct": False, "reason": "whole answer is not JSON"}
    correct = isinstance(parsed, dict) and parsed == {"answer": gold}
    return {"correct": correct, "reason": "exact structured answer" if correct else "wrong answer/schema"}


class ReferenceSuite(Suite):
    def server_command(self):
        a = self.args
        command = [str(a.binary.resolve()), "-m", str(a.model.resolve()),
                   "--host", "127.0.0.1", "--port", str(a.port), "-c", str(a.context),
                   "-b", str(a.chunk), "-ngl", "99", "--kv-dtype", "q8_0",
                   "--no-ui", "--no-think", "--temp", "0", "--top-k", "0",
                   "--top-p", "1", "--presence-penalty", "0", "--frequency-penalty", "0",
                   "--repeat-penalty", "1", "--seed", "74191"]
        if a.window == 0:
            command += ["--no-kvmem"]
        else:
            command += ["--kvmem", "--kvmem-budget", str(a.reference_budget),
                        "--kvmem-gen-reserve", str(a.reference_reserve),
                        "--kvmem-cpu-gb", str(a.host_mib / 1024),
                        "--kvmem-block-tokens", "128", "--kvmem-method", "retrieval",
                        "--kvmem-query-max-tokens", "512", "--kvmem-query-replay", "auto",
                        "--kvmem-query-policy", "user"]
        command += ["--spec-type", "draft-mtp" if a.spec == "mtp" else "none"]
        if a.spec == "mtp":
            command += ["--spec-draft-n-max", "3", "--spec-kv-dtype", "f16",
                        "--kvmem-mtp-state", "replay"]
        return command


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_bytes = args.fixtures.read_bytes()
    dataset = json.loads(fixture_bytes)
    (args.output / "fixtures.json").write_bytes(fixture_bytes)
    report = {"kind": "diagnostic-only", "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": [], "complete": False}
    def save():
        (args.output / "quality.json").write_text(encode(report), encoding="utf-8")
    # Reuse the smoke runner's bounded child ownership, readiness and GPU monitoring.
    args.profile = "smoke"
    suite = ReferenceSuite(args) if args.engine == "kvmem" else Suite(args)
    save()
    try:
        report["server"] = suite.start()
        if args.audit_sampling_history:
            report["probe_sampling_audit"] = suite.replay_sampling_history()
        for case in dataset["cases"]:
            start = time.monotonic()
            row = {"id": case["id"], "family": case["family"], "gold": case["gold"]}
            request = {"model": "kvmem-test", "messages": case["messages"], "max_tokens": 128,
                       "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}
            if args.presence_penalty:
                request["presence_penalty"] = args.presence_penalty
            if args.frequency_penalty:
                request["frequency_penalty"] = args.frequency_penalty
            row["request"] = request
            try:
                req = urllib.request.Request(suite.url + "/v1/chat/completions",
                    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
                    row["response"] = json.load(response)
                validate_json(row["response"])
                row["grade"] = grade(row["response"], case["gold"])
            except Exception:
                row["error"] = traceback.format_exc()
                row["grade"] = {"correct": False, "reason": "request failed"}
            row["seconds"] = time.monotonic() - start
            report["cases"].append(row)
            save()
            print(f"{case['id']}: {row['grade']} ({row['seconds']:.1f}s)", flush=True)
        report["complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            suite.stop()
            report["engine_log"] = suite.check_log()
        except Exception:
            report["shutdown_error"] = traceback.format_exc()
        report["correct"] = sum(row["grade"]["correct"] for row in report["cases"])
        report["total"] = len(dataset["cases"])
        save()
    return int(not report["complete"] or "error" in report or "shutdown_error" in report
               or report["correct"] != report["total"])


def compare(paths, cluster_map=None):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({r["fixture_sha256"] for r in reports}) != 1:
        raise ValueError("paired comparison requires identical frozen fixtures")
    baseline = reports[0]
    expected_ids = [r["id"] for r in baseline["cases"]]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate case IDs")
    semantic = any("review" in report or report.get("kind", "").startswith("longmemeval")
                   for report in reports)
    for report in reports:
        if (not report["complete"] or len(report["cases"]) != report["total"] or
                [r["id"] for r in report["cases"]] != expected_ids or
                "error" in report or "shutdown_error" in report or report.get("request_errors") or
                any("error" in row for row in report["cases"])):
            raise ValueError("incomplete or failed run cannot establish quality")
        labels = [row.get("grade", {}).get("correct") for row in report["cases"]]
        if any(type(label) is not bool for label in labels) or sum(labels) != report["correct"]:
            raise ValueError("missing labels or inconsistent correctness total")
        if semantic:
            review = report.get("review", {})
            if not report.get("scored") or not review.get("reviewer") or not review.get("rubric"):
                raise ValueError("semantic comparison requires completed identified reviews")
            for field in ("reviewer", "rubric", "judge_settings", "adjudication"):
                if review.get(field) != baseline.get("review", {}).get(field):
                    raise ValueError(f"semantic comparison changed {field}")
    cluster_by_id = None
    if baseline.get('kind', '').startswith('longmemeval') and cluster_map is None:
        raise ValueError('LongMemEval comparison requires the frozen dependence-cluster map')
    if cluster_map is not None:
        if (cluster_map.get('kind') != 'longmemeval-question-clusters' or
                any(report.get('source', {}).get('sha256') != cluster_map.get('source_sha256')
                    for report in reports)):
            raise ValueError('cluster map does not match prediction source')
        cluster_by_id = {}
        seen_clusters = set()
        for group in cluster_map['clusters']:
            if not group['question_ids'] or group['id'] in seen_clusters:
                raise ValueError('empty or duplicate dependence cluster')
            seen_clusters.add(group['id'])
            for qid in group['question_ids']:
                if qid in cluster_by_id:
                    raise ValueError('question appears in more than one cluster')
                cluster_by_id[qid] = group['id']
        if (len(cluster_by_id) != cluster_map['question_count'] or
                any(qid not in cluster_by_id for qid in expected_ids)):
            raise ValueError('dependence map is incomplete')
    result = {"kind": baseline.get("kind", "diagnostic-only"), "equivalence_established": False,
              "noninferiority_margin": 0.01, "comparisons": []}
    if cluster_map is not None:
        result['clustering'] = {'sha256': hashlib.sha256(encode(cluster_map).encode()).hexdigest(),
                                'rule': cluster_map['rule']}
    for path, candidate in zip(paths[1:], reports[1:]):
        regressions, improvements = [], []
        for left, right in zip(baseline["cases"], candidate["cases"]):
            a, b = left["grade"]["correct"], right["grade"]["correct"]
            if a and not b:
                regressions.append(left["id"])
            if b and not a:
                improvements.append(left["id"])
        def statistics(pairs):
            a_labels = [a["grade"]["correct"] for a, _ in pairs]
            b_labels = [b["grade"]["correct"] for _, b in pairs]
            stats = (paired_cluster_accuracy(a_labels, b_labels, [cluster_by_id[a['id']] for a, _ in pairs])
                     if cluster_by_id is not None else paired_accuracy(a_labels, b_labels))
            stats["interval_supports_1pp_noninferiority"] = stats["difference_interval"][0] >= -0.01
            return stats
        pairs = list(zip(baseline["cases"], candidate["cases"]))
        families = sorted({a.get("family", "unspecified") for a, _ in pairs})
        result["comparisons"].append({"candidate": str(path), "baseline_correct": baseline["correct"],
            "candidate_correct": candidate["correct"], "total": candidate["total"],
            "regressions": regressions, "improvements": improvements,
            "paired_statistics": statistics(pairs),
            "by_family": {family: statistics([(a, b) for a, b in pairs
                          if a.get("family", "unspecified") == family]) for family in families}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("fixtures")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--rows", type=int, default=400)
    create.add_argument("--seed", type=int, default=74191)
    execute = sub.add_parser("run")
    execute.add_argument("--fixtures", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--binary", type=Path, default=Path("build/apps/ninfer-serve"))
    execute.add_argument("--engine", choices=("ninfer", "kvmem"), default="ninfer")
    execute.add_argument("--reference-budget", type=int)
    execute.add_argument("--reference-reserve", type=int)
    execute.add_argument("--model", type=Path, required=True)
    execute.add_argument("--port", type=int, default=8105)
    execute.add_argument("--context", type=int, default=32768)
    execute.add_argument("--window", type=int, default=64)
    execute.add_argument("--chunk", type=int, default=1024)
    execute.add_argument("--host-mib", type=int, default=4096)
    execute.add_argument("--dtype", default="int8")
    execute.add_argument("--spec", choices=("none", "mtp"), default="none")
    execute.add_argument("--presence-penalty", type=float, default=0)
    execute.add_argument("--frequency-penalty", type=float, default=0)
    execute.add_argument("--audit-sampling-history", action="store_true")
    execute.add_argument("--startup-timeout", type=float, default=600)
    execute.add_argument("--request-timeout", type=float, default=1800)
    paired = sub.add_parser("compare")
    paired.add_argument("reports", type=Path, nargs="+")
    paired.add_argument("--clusters", type=Path)
    args = parser.parse_args()
    if args.command == "fixtures":
        if args.rows < 20:
            parser.error("at least 20 records are required")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(encode(fixtures(args.rows, args.seed)))
        return 0
    if args.command == "compare":
        print(encode(compare(args.reports, json.loads(args.clusters.read_bytes()) if args.clusters else None)))
        return 0
    if args.engine == "kvmem" and args.window and (
            args.reference_budget is None or args.reference_reserve is None):
        parser.error("set explicit --reference-budget and --reference-reserve for KVMem")
    if args.audit_sampling_history and (args.engine != "ninfer" or not args.window):
        parser.error("sampling history audit requires sparse NInfer")
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
