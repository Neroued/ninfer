"""Compare two structured activation dumps without imposing token equality."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from tools.q5090.ref.config import CFG


def load_manifest(root: Path) -> dict:
    with (root / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != "qus_activation_dump_v1":
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
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return records(load_manifest(root), phase)
    # C++ FileTap currently writes one flat phase/step without a manifest.
    # Infer only the stable layer-level contract; op-level dumps require a
    # manifest because their shapes are not derivable from filenames.
    out = {}
    patterns = {
        "embed.f32": "embed",
        "final_norm.f32": "final_norm",
        "logits.f32": "logits",
    }
    for path in sorted(root.glob("*.f32")):
        name = patterns.get(path.name)
        if name is None:
            match = re.fullmatch(r"layer_(\d{2})_(mixer|mlp)\.f32", path.name)
            if match:
                name = f"layer_{int(match.group(1)):02d}/{match.group(2)}"
        if name is None:
            raise ValueError(f"{root}: cannot infer legacy C++ tap {path.name}")
        elements = path.stat().st_size // np.dtype(np.float32).itemsize
        width = CFG.vocab if name == "logits" else CFG.hidden
        if elements % width:
            raise ValueError(f"{path}: file size is not divisible by inferred width {width}")
        shape = [elements // width, width]
        key = (phase, 0, 0, name)
        out[key] = {"file": path.name, "shape": shape}
    if not out:
        raise ValueError(f"{root}: no activation dump files")
    return out


def metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(diff), initial=0.0))
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    an = float(np.linalg.norm(a.astype(np.float64)))
    bn = float(np.linalg.norm(b.astype(np.float64)))
    cosine = 1.0 if an == 0 and bn == 0 else 0.0 if an == 0 or bn == 0 else float(
        np.dot(a.astype(np.float64), b.astype(np.float64)) / (an * bn)
    )
    return max_abs, rms, cosine


def compare(
    reference: Path,
    candidate: Path,
    *,
    reference_phase: str = "prefill",
    candidate_phase: str = "prefill",
) -> list[dict]:
    ref = load_records(reference, reference_phase)
    got = load_records(candidate, candidate_phase)
    report = []
    for key in sorted(ref.keys() | got.keys()):
        if key not in ref or key not in got:
            report.append(
                {
                    "key": key,
                    "status": "missing",
                    "side": "reference" if key not in ref else "candidate",
                }
            )
            continue
        ra, rb = ref[key], got[key]
        if ra["shape"] != rb["shape"]:
            report.append(
                {
                    "key": key,
                    "status": "shape",
                    "reference": ra["shape"],
                    "candidate": rb["shape"],
                }
            )
            continue
        a = np.fromfile(reference / ra["file"], dtype=np.float32)
        b = np.fromfile(candidate / rb["file"], dtype=np.float32)
        if a.size != b.size:
            report.append(
                {
                    "key": key,
                    "status": "size",
                    "reference": int(a.size),
                    "candidate": int(b.size),
                }
            )
            continue
        max_abs, rms, cosine = metrics(a, b)
        report.append(
            {
                "key": key,
                "status": "ok",
                "max_abs": max_abs,
                "rms": rms,
                "cosine": cosine,
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
    for item in report:
        key = "/".join(map(str, item["key"]))
        if item["status"] == "ok":
            print(
                f"{key}: max_abs={item['max_abs']:.7g} "
                f"rms={item['rms']:.7g} cosine={item['cosine']:.9f}"
            )
        else:
            print(f"{key}: {item['status']} {item}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
