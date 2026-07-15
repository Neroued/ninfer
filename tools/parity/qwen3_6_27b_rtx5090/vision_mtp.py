"""Run the sequential Python/C++ multimodal Vision and MTP parity gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.reference.qwen3_6.common.frontend import Frontend
from tools.reference.qwen3_6.common.multimodal import load_messages
from tools.reference.qwen3_6.common.tap import FileTap
from tools.reference.qwen3_6_27b_rtx5090 import RefModel


# These are cross-runtime tolerances for the same quantized artifact, not comparisons with BF16.
# Each rule requires both a bounded relative RMS error and a retained cosine similarity.
TOLERANCES: dict[str, dict[str, float]] = {
    "vision/patch_embed": {"max_relative_rmse": 0.03, "min_cosine": 0.999},
    "vision/block_00": {"max_relative_rmse": 0.06, "min_cosine": 0.995},
    "vision/block_13": {"max_relative_rmse": 0.18, "min_cosine": 0.97},
    "vision/block_26": {"max_relative_rmse": 0.25, "min_cosine": 0.94},
    "vision/merger": {"max_relative_rmse": 0.25, "min_cosine": 0.94},
    "embed": {"max_relative_rmse": 0.25, "min_cosine": 0.94},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cpp", required=True, help="ninfer-qwen3_6_27b-dump executable")
    parser.add_argument("--messages", required=True, help="mixed image/video messages JSON")
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--mtp-draft-tokens", type=int, default=1)
    parser.add_argument("--proposal-head", choices=("full", "optimized"), default="full")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _frontend_metadata(batch) -> dict[str, Any]:
    return {
        "token_ids": batch.input_ids.tolist(),
        "token_types": batch.mm_token_type_ids.tolist(),
        "position_ids": batch.position_ids.tolist(),
        "rope_delta": batch.rope_delta,
    }


def _speculative_metadata(model: RefModel) -> dict[str, Any]:
    stats = model.mtp_stats
    return {
        "enabled": model.mtp_enabled,
        "draft_window": model.mtp_draft_tokens,
        "rounds": stats.rounds,
        "drafted_tokens": stats.draft_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "fallback_steps": stats.fallback_steps,
        "accepted_per_position": stats.accepted_per_pos[: model.mtp_draft_tokens],
    }


def _run_python(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    messages = load_messages(args.messages)
    tap = FileTap(output, "layer")
    device = torch.device(f"cuda:{args.device}")
    with RefModel(
        args.weights,
        device=device,
        kv_dtype=args.kv_dtype,
        prefill_chunk=args.prefill_chunk,
        mtp_draft_tokens=args.mtp_draft_tokens,
        draft_head=args.proposal_head == "optimized",
        compile_codec=False,
    ) as model, torch.inference_mode():
        frontend = Frontend(model.binding)
        batch = frontend.process(messages, thinking=args.thinking)
        if batch.image_tokens == 0 or batch.video_tokens == 0:
            raise ValueError("--messages must contain both image and video content")

        def vision_tap(name: str, value: torch.Tensor) -> None:
            tap(
                f"vision/{name}",
                value,
                phase="prefill",
                step=0,
                chunk=0,
                position_begin=0,
            )

        class ComposedEmbeddingTap:
            level = "layer"

            def __call__(self, name: str, value: torch.Tensor, **context: Any) -> None:
                if (
                    name == "embed"
                    and context["phase"] == "prefill"
                    and context["chunk"] == 0
                ):
                    tap(name, value, **context)

        vision = model.encode_vision(batch, compile_codec=False, tap=vision_tap)
        generated = model.generate_multimodal(batch, vision, 3, tap=ComposedEmbeddingTap())
        metadata = {
            "runtime": "python-ninfer",
            "weights": str(Path(args.weights).resolve()),
            "prompt_source": "messages",
            "messages": str(Path(args.messages).resolve()),
            "frontend": _frontend_metadata(batch),
            "generated_token_ids": generated,
            "mtp_draft_tokens": args.mtp_draft_tokens,
            "draft_head": args.proposal_head,
            "speculative": _speculative_metadata(model),
            "vision": True,
        }
        tap.close(**metadata)
        return metadata


def _run_cpp(args: argparse.Namespace, output: Path) -> dict[str, str]:
    command = [
        str(Path(args.cpp)),
        "--weights",
        str(Path(args.weights)),
        "--messages",
        str(Path(args.messages)),
        "--decode",
        "3",
        "--greedy",
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--kv-dtype",
        args.kv_dtype,
        "--mtp-draft-tokens",
        str(args.mtp_draft_tokens),
        "--proposal-head",
        args.proposal_head,
        "--activation-dump",
        str(output),
        "--dump-level",
        "vision-mtp",
        "--device",
        str(args.device),
    ]
    if not args.thinking:
        command.append("--no-thinking")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "C++ diagnostic failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return {"stdout": completed.stdout, "stderr": completed.stderr}


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing activation manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "ninfer_activation_dump_v1":
        raise ValueError(f"unsupported activation manifest: {path}")
    return value


def _check_frontend(reference: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for field in ("token_ids", "token_types", "position_ids", "rope_delta"):
        expected = reference.get(field)
        actual = candidate.get(field)
        passed = expected == actual
        item: dict[str, Any] = {"field": field, "status": "ok" if passed else "mismatch"}
        if not passed:
            if isinstance(expected, list) and isinstance(actual, list):
                item["reference_shape"] = list(np.asarray(expected).shape)
                item["candidate_shape"] = list(np.asarray(actual).shape)
                if np.asarray(expected).shape == np.asarray(actual).shape:
                    lhs = np.asarray(expected)
                    rhs = np.asarray(actual)
                    differing = np.flatnonzero(lhs.reshape(-1) != rhs.reshape(-1))
                    item["first_difference"] = None if not differing.size else int(differing[0])
            else:
                item["reference"] = expected
                item["candidate"] = actual
        results.append(item)
    return results


def _records(manifest: dict[str, Any]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result = {}
    for record in manifest.get("tensors", []):
        key = (
            record.get("name"),
            record.get("phase"),
            int(record.get("step", 0)),
            int(record.get("chunk", 0)),
        )
        if key in result:
            raise ValueError(f"duplicate activation record: {key}")
        result[key] = record
    return result


def _activation_result(
    name: str,
    reference_root: Path,
    candidate_root: Path,
    reference: dict[tuple[str, str, int, int], dict[str, Any]],
    candidate: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (name, "prefill", 0, 0)
    tolerance = TOLERANCES[name]
    result: dict[str, Any] = {"name": name, "tolerance": tolerance}
    if key not in reference or key not in candidate:
        result["status"] = "missing"
        result["missing"] = [
            side
            for side, records in (("reference", reference), ("candidate", candidate))
            if key not in records
        ]
        return result
    expected_record = reference[key]
    actual_record = candidate[key]
    if expected_record.get("shape") != actual_record.get("shape"):
        result.update(
            status="shape",
            reference_shape=expected_record.get("shape"),
            candidate_shape=actual_record.get("shape"),
        )
        return result
    expected = np.fromfile(reference_root / expected_record["file"], dtype=np.float32)
    actual = np.fromfile(candidate_root / actual_record["file"], dtype=np.float32)
    elements = math.prod(expected_record["shape"])
    if expected.size != elements or actual.size != elements:
        result.update(
            status="shape",
            expected_elements=elements,
            reference_elements=int(expected.size),
            candidate_elements=int(actual.size),
        )
        return result
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        result.update(
            status="non-finite",
            reference_non_finite=int((~np.isfinite(expected)).sum()),
            candidate_non_finite=int((~np.isfinite(actual)).sum()),
        )
        return result

    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    reference_rms = (
        float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))) if expected.size else 0.0
    )
    relative_rmse = 0.0 if reference_rms == 0.0 and rms == 0.0 else math.inf
    if reference_rms != 0.0:
        relative_rmse = rms / reference_rms
    expected_norm = float(np.linalg.norm(expected.astype(np.float64)))
    actual_norm = float(np.linalg.norm(actual.astype(np.float64)))
    if expected_norm == 0.0 and actual_norm == 0.0:
        cosine = 1.0
    elif expected_norm == 0.0 or actual_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            np.dot(expected.astype(np.float64), actual.astype(np.float64))
            / (expected_norm * actual_norm)
        )
    passed = (
        relative_rmse <= tolerance["max_relative_rmse"]
        and cosine >= tolerance["min_cosine"]
    )
    result.update(
        status="ok" if passed else "tolerance",
        max_abs=float(np.max(np.abs(delta), initial=0.0)),
        rms=rms,
        relative_rmse=relative_rmse,
        cosine=cosine,
    )
    return result


def _check_mtp(manifest: dict[str, Any], runtime: str) -> dict[str, Any]:
    stats = manifest.get("speculative", {})
    checks = {
        "configured_k1": manifest.get("mtp_draft_tokens") == 1,
        "full_head": manifest.get("draft_head") == "full",
        "enabled": stats.get("enabled") is True,
        "draft_window_k1": stats.get("draft_window") == 1,
        "round_executed": isinstance(stats.get("rounds"), int) and stats["rounds"] >= 1,
        "proposal_executed": isinstance(stats.get("drafted_tokens"), int)
        and stats["drafted_tokens"] >= 1,
    }
    return {"runtime": runtime, "status": "ok" if all(checks.values()) else "failed", **checks}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.prefill_chunk <= 0 or args.prefill_chunk % 128:
        parser.error("--prefill-chunk must be a nonzero multiple of 128")
    if args.mtp_draft_tokens != 1 or args.proposal_head != "full":
        parser.error("this fixed parity probe requires --mtp-draft-tokens 1 --proposal-head full")

    work = Path(args.work_dir)
    python_root = work / "python"
    cpp_root = work / "cpp"
    work.mkdir(parents=True, exist_ok=True)
    python_root.mkdir(parents=True, exist_ok=True)
    cpp_root.mkdir(parents=True, exist_ok=True)

    # The Python reference is deliberately complete and released before the C++ process starts.
    _run_python(args, python_root)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cpp_process = _run_cpp(args, cpp_root)

    python_manifest = _load_manifest(python_root)
    cpp_manifest = _load_manifest(cpp_root)
    frontend = _check_frontend(
        python_manifest.get("frontend", {}), cpp_manifest.get("frontend", {})
    )
    reference_records = _records(python_manifest)
    candidate_records = _records(cpp_manifest)
    activations = [
        _activation_result(
            name,
            python_root,
            cpp_root,
            reference_records,
            candidate_records,
        )
        for name in TOLERANCES
    ]
    mtp = [_check_mtp(python_manifest, "python"), _check_mtp(cpp_manifest, "cpp")]
    passed = (
        all(item["status"] == "ok" for item in frontend)
        and all(item["status"] == "ok" for item in activations)
        and all(item["status"] == "ok" for item in mtp)
    )
    report = {
        "format": "ninfer_qwen3_6_27b_vision_mtp_parity_v1",
        "status": "ok" if passed else "failed",
        "weights": str(Path(args.weights).resolve()),
        "messages": str(Path(args.messages).resolve()),
        "frontend": frontend,
        "activations": activations,
        "mtp": mtp,
        "cpp_stdout": cpp_process["stdout"],
        "cpp_stderr": cpp_process["stderr"],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
