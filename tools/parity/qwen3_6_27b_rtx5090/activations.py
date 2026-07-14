"""Gate two structured activation dumps without imposing final-token equality."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from tools.reference.qwen3_6_27b_rtx5090.config import CFG


# Reviewed cross-runtime rules for the same quantized artifact. They cover every layer-level Text
# tap emitted by both implementations and deliberately do not depend on layer number or run output.
TOLERANCE_RULES: tuple[tuple[str, float, float], ...] = (
    (r"embed", 0.01, 0.9999),
    (r"layer_[0-9]{2}/mixer", 0.25, 0.94),
    (r"layer_[0-9]{2}/mlp", 0.25, 0.94),
    (r"final_norm", 0.25, 0.94),
    (r"logits", 0.35, 0.90),
)


def load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing activation manifest: {path}")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != "ninfer_activation_dump_v1":
        raise ValueError(f"{root}: unsupported activation dump format")
    return manifest


def records(manifest: dict, phase: str | None = None) -> dict[tuple, dict]:
    out = {}
    for record in manifest.get("tensors", []):
        if phase is not None and record.get("phase") != phase:
            continue
        if record.get("name") == "logits" and record.get("shape") == [CFG.vocab]:
            record = dict(record)
            record["shape"] = [1, CFG.vocab]
        key = (
            record["phase"],
            record["step"],
            record["chunk"],
            record["name"],
        )
        if key in out:
            raise ValueError(f"duplicate activation record {key}")
        out[key] = record
    return out


def load_records(root: Path, phase: str) -> dict[tuple, dict]:
    return records(load_manifest(root), phase)


def tolerance(name: str) -> dict[str, float] | None:
    for pattern, max_relative_rmse, min_cosine in TOLERANCE_RULES:
        if re.fullmatch(pattern, name):
            return {
                "max_relative_rmse": max_relative_rmse,
                "min_cosine": min_cosine,
            }
    return None


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    diff = af - bf
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    reference_rms = float(np.sqrt(np.mean(af * af))) if af.size else 0.0
    if reference_rms == 0.0:
        relative_rmse = 0.0 if rms == 0.0 else math.inf
    else:
        relative_rmse = rms / reference_rms
    an = float(np.linalg.norm(af))
    bn = float(np.linalg.norm(bf))
    cosine = (
        1.0
        if an == 0.0 and bn == 0.0
        else 0.0
        if an == 0.0 or bn == 0.0
        else float(np.dot(af, bf) / (an * bn))
    )
    return {
        "max_abs": float(np.max(np.abs(diff), initial=0.0)),
        "rms": rms,
        "relative_rmse": relative_rmse,
        "cosine": cosine,
    }


def compare(
    reference: Path,
    candidate: Path,
    *,
    reference_phase: str = "prefill",
    candidate_phase: str = "prefill",
) -> list[dict]:
    ref = load_records(reference, reference_phase)
    got = load_records(candidate, candidate_phase)
    report: list[dict] = []
    if not ref:
        report.append(
            {
                "key": [reference_phase, 0, 0, "*"],
                "status": "missing_phase",
                "side": "reference",
            }
        )
    if not got:
        report.append(
            {
                "key": [candidate_phase, 0, 0, "*"],
                "status": "missing_phase",
                "side": "candidate",
            }
        )
    if not ref or not got:
        return report

    for key in sorted(ref.keys() | got.keys()):
        display_key = list(key)
        if key not in ref:
            report.append({"key": display_key, "status": "unexpected"})
            continue
        if key not in got:
            report.append({"key": display_key, "status": "missing"})
            continue
        expected_record, actual_record = ref[key], got[key]
        if expected_record["shape"] != actual_record["shape"]:
            report.append(
                {
                    "key": display_key,
                    "status": "shape",
                    "reference": expected_record["shape"],
                    "candidate": actual_record["shape"],
                }
            )
            continue
        expected_path = reference / expected_record["file"]
        actual_path = candidate / actual_record["file"]
        missing_files = [
            side
            for side, path in (("reference", expected_path), ("candidate", actual_path))
            if not path.is_file()
        ]
        if missing_files:
            report.append(
                {"key": display_key, "status": "missing_file", "side": missing_files}
            )
            continue
        expected = np.fromfile(expected_path, dtype=np.float32)
        actual = np.fromfile(actual_path, dtype=np.float32)
        elements = math.prod(expected_record["shape"])
        if expected.size != elements or actual.size != elements:
            report.append(
                {
                    "key": display_key,
                    "status": "size",
                    "shape_elements": elements,
                    "reference": int(expected.size),
                    "candidate": int(actual.size),
                }
            )
            continue
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            report.append(
                {
                    "key": display_key,
                    "status": "non_finite",
                    "reference": int((~np.isfinite(expected)).sum()),
                    "candidate": int((~np.isfinite(actual)).sum()),
                }
            )
            continue
        rule = tolerance(key[3])
        if rule is None:
            report.append({"key": display_key, "status": "missing_tolerance"})
            continue
        measured = metrics(expected, actual)
        passed = (
            measured["relative_rmse"] <= rule["max_relative_rmse"]
            and measured["cosine"] >= rule["min_cosine"]
        )
        report.append(
            {
                "key": display_key,
                "status": "ok" if passed else "tolerance",
                **measured,
                "tolerance": rule,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--reference-phase", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--candidate-phase", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--json")
    args = parser.parse_args()
    report = compare(
        Path(args.reference),
        Path(args.candidate),
        reference_phase=args.reference_phase,
        candidate_phase=args.candidate_phase,
    )
    passed = bool(report) and all(item["status"] == "ok" for item in report)
    for item in report:
        key = "/".join(map(str, item["key"]))
        if item["status"] == "ok":
            print(
                f"{key}: max_abs={item['max_abs']:.7g} rms={item['rms']:.7g} "
                f"relative_rmse={item['relative_rmse']:.7g} cosine={item['cosine']:.9f}"
            )
        else:
            print(f"{key}: {item['status']} {item}")
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
