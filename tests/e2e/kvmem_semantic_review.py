#!/usr/bin/env python3
"""Validate and attach explicit whole-answer reviews without changing raw predictions.

Produces an unscored review template, or imports a completed review. It makes no
judge API calls. Review labels must name the reviewer and rubric and remain bound
to the exact prediction report. Manual rubric interpretation is not an official
GPT-4o evaluator result. A graded pilot is still not equivalence evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kvmem_quality import encode


def review_template(raw):
    report = json.loads(raw)
    if (not report.get("complete") or report.get("request_errors") or
            "error" in report or "shutdown_error" in report or
            len(report["cases"]) != report["total"]):
        raise ValueError("prediction run is incomplete or contains execution errors")
    return {"prediction_sha256": hashlib.sha256(raw).hexdigest(),
            "fixture_sha256": report["fixture_sha256"], "reviewer": "", "rubric": "",
            "instructions": (
                "Review the complete answer against the task-specific LongMemEval rubric. "
                "Also label strict correctness: no missing required facts, contradictory "
                "reasoning, or material unsupported assertions. A correct substring alone "
                "does not pass. Retain failures; mark uncertain cases unresolved (null). "
                "Record rationale and evidence for both labels. Do not infer missing labels."),
            "cases": [{"id": row["id"], "family": row["family"],
                       "question": row["question"], "gold": row["gold"],
                       "answer": row["response"]["choices"][0]["message"]["content"],
                       "rubric_correct": None, "strict_correct": None, "rationale": ""}
                      for row in report["cases"]]}


def attach(raw, review):
    template = review_template(raw)
    for field in ("prediction_sha256", "fixture_sha256"):
        if review.get(field) != template[field]:
            raise ValueError(f"{field} does not match")
    for field in ("reviewer", "rubric"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"missing {field}")
    expected = template["cases"]
    rows = review["cases"]
    if len(rows) != len(expected) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("missing or duplicate review IDs")
    for actual, original in zip(rows, expected):
        for key in ("id", "family", "question", "gold", "answer"):
            if actual[key] != original[key]:
                raise ValueError("review changed or reordered the frozen evaluation input")
        if any(type(actual.get(key)) is not bool for key in ("rubric_correct", "strict_correct")):
            raise ValueError("unresolved or nonboolean review label")
        if not isinstance(actual.get("rationale"), str) or not actual["rationale"].strip():
            raise ValueError("missing review rationale")
        if actual["strict_correct"] and not actual["rubric_correct"]:
            raise ValueError("strict pass cannot override failed base rubric")
    report = json.loads(raw)
    for row, label in zip(report["cases"], rows):
        row["grade"] = {"correct": label["strict_correct"], "reason": label["rationale"]}
        for field in ("evidence", "evidence_source_corrections"):
            if field in label:
                row["grade"][field] = label[field]
        row["rubric_correct"] = label["rubric_correct"]
    report["correct"] = sum(r["strict_correct"] for r in rows)
    report["rubric_correct"] = sum(r["rubric_correct"] for r in rows)
    report["scored"] = True
    report["review"] = {k: review[k] for k in ("prediction_sha256", "reviewer", "rubric")}
    for field in ("judge_report_sha256", "judge_settings", "adjudication"):
        if field in review:
            report["review"][field] = review[field]
    # Bind the entire canonical review, including evidence and explicit corrections,
    # rather than retaining only the reviewer name after source adjudication.
    report["review"]["review_record_sha256"] = hashlib.sha256(encode(review).encode()).hexdigest()
    report["equivalence_established"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, help="omit to create an unscored review template")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.predictions.read_bytes()
    result = attach(raw, json.loads(args.reviews.read_bytes())) if args.reviews else review_template(raw)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(encode(result))


if __name__ == "__main__":
    main()
