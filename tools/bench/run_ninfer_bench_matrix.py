#!/usr/bin/env python3
"""Run the native NInfer product performance matrix.

The matrix is intentionally layered instead of fully factorial:

* k=3 is the primary MTP path to evaluate.
* k=0 and k=5 are baseline/max-window controls.
* k=0..5 is swept on representative context-decode cases.
* CUDA graph is compared only for decode-bearing tests.
* Prefill-only tests sweep length and chunk size, but not graph on/off.

Raw ninfer_bench reports stay under profiles/bench. This script writes a
descriptive manifest, exact commands, per-case logs, raw JSON reports, and a flat
summary CSV/JSON that is easy to compare across runs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO_ROOT / "build/bench/ninfer_bench"
DEFAULT_WEIGHTS = REPO_ROOT / "out/qwen3_6_27b_rtx5090.ninfer"
DEFAULT_CORPUS = REPO_ROOT / "bench/fixtures/bench_corpus.ids"

PREFILL_LENGTHS_CORE = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PREFILL_LENGTHS_FULL_EXTRA = (32768, 65536)
PREFILL_CHUNKS = (128, 256, 512, 1024, 2048, 4096)
PURE_DECODE_GENS = (16, 64, 128, 512, 2048)
CONTEXT_CORE = ((512, 512), (2048, 512), (8192, 512))
CONTEXT_FULL_EXTRA = ((32768, 256), (65536, 128))
PRIMARY_KS = (0, 3, 5)
SWEEP_KS = (0, 1, 2, 3, 4, 5)
REPORT_SCHEMA_VERSION = 8
REPORT_ARTIFACT_TYPE = "ninfer_bench_report"
REPORT_TOOL = "ninfer_bench"


@dataclasses.dataclass(frozen=True)
class BenchCase:
    suite: str
    name: str
    args: tuple[str, ...]
    repetitions: int
    warmup: int
    notes: str = ""


def csv_list(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def pair_list(values: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{p},{g}" for p, g in values)


def mtp_args(k: int) -> tuple[str, ...]:
    args = ("--mtp-draft-tokens", str(k))
    return (*args, "--lm-head-draft") if k > 0 else args


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def count_corpus_tokens(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"corpus file not found: {path}")
    return len(path.read_text(encoding="utf-8").split())


def add_repetition_args(
    base_args: list[str], case: BenchCase, repetitions_override: int | None,
    warmup_override: int | None,
) -> list[str]:
    repetitions = repetitions_override if repetitions_override is not None else case.repetitions
    warmup = warmup_override if warmup_override is not None else case.warmup
    return [*base_args, "-r", str(repetitions), "--warmup", str(warmup)]


def build_cases(preset: str) -> list[BenchCase]:
    if preset == "smoke":
        return [
            BenchCase("prefill_length", "prefill_p128_k0", ("-p", "128", *mtp_args(0)), 1, 0),
            BenchCase("pure_decode", "tg8_k3_graph", ("-n", "8", *mtp_args(3)), 1, 0),
            BenchCase(
                "context_decode",
                "ctx_p128_g8_k3_graph",
                ("-pg", "128,8", "--max-ctx", "256", *mtp_args(3)),
                1,
                0,
            ),
        ]

    include_full = preset == "full"
    prefill_lengths = PREFILL_LENGTHS_CORE + (PREFILL_LENGTHS_FULL_EXTRA if include_full else ())
    context_pairs = CONTEXT_CORE + (CONTEXT_FULL_EXTRA if include_full else ())
    sweep_pairs = ((2048, 512),) + (((32768, 256),) if include_full else ())

    cases: list[BenchCase] = []

    for k in PRIMARY_KS:
        cases.append(
            BenchCase(
                "prefill_length",
                f"prefill_lengths_k{k}",
                ("-p", csv_list(prefill_lengths), *mtp_args(k)),
                3,
                1,
                "prefill length curve",
            )
        )

    for k in PRIMARY_KS:
        for chunk in PREFILL_CHUNKS:
            cases.append(
                BenchCase(
                    "prefill_chunk",
                    f"prefill_p8192_chunk{chunk}_k{k}",
                    (
                        "-p",
                        "8192",
                        "--prefill-chunk",
                        str(chunk),
                        *mtp_args(k),
                    ),
                    3,
                    1,
                    "workspace and chunk-size sensitivity",
                )
            )

    for k in PRIMARY_KS:
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = ["-n", csv_list(PURE_DECODE_GENS), *mtp_args(k)]
            if not graph:
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "pure_decode",
                    f"tg_lengths_k{k}_{graph_suffix}",
                    tuple(args),
                    5,
                    1,
                    "pure decode throughput; tg seeds only one token",
                )
            )

    for k in PRIMARY_KS:
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = ["-pg", pair_list(context_pairs), *mtp_args(k)]
            if not graph:
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "context_decode",
                    f"context_decode_k{k}_{graph_suffix}",
                    tuple(args),
                    3,
                    1,
                    "decode at real context offsets",
                )
            )

    for k in SWEEP_KS:
        cases.append(
            BenchCase(
                "mtp_sweep",
                f"mtp_sweep_k{k}_graph",
                ("-pg", pair_list(sweep_pairs), *mtp_args(k)),
                3,
                1,
                "primary MTP draft-window sweep",
            )
        )

    for k, prompt in ((3, 8174), (5, 8170)):
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = [
                "-pg",
                f"{prompt},12",
                "--max-ctx",
                "8192",
                *mtp_args(k),
            ]
            if not graph:
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "tail_stress",
                    f"tail_k{k}_{graph_suffix}",
                    tuple(args),
                    3,
                    1,
                    "near-capacity fallback stress",
                )
            )

    return cases


def filtered_cases(cases: list[BenchCase], suites: Sequence[str], limit: int | None) -> list[BenchCase]:
    selected = cases
    if suites:
        allowed = set(suites)
        selected = [case for case in selected if case.suite in allowed]
    if limit is not None:
        selected = selected[:limit]
    return selected


def max_prompt_in_cases(cases: Sequence[BenchCase]) -> int:
    max_prompt = 0
    for case in cases:
        args = list(case.args)
        for flag in ("-p", "--n-prompt"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                max_prompt = max(max_prompt, *(int(piece) for piece in raw.split(",")))
        for flag in ("-pg", "--prompt-gen"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                for pair in raw.split(";"):
                    p, _ = pair.split(",", 1)
                    max_prompt = max(max_prompt, int(p))
    return max_prompt


def load_bench_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("benchmark report root must be an object")
    identity = (
        report.get("schema_version"),
        report.get("artifact_type"),
        report.get("tool"),
    )
    expected = (REPORT_SCHEMA_VERSION, REPORT_ARTIFACT_TYPE, REPORT_TOOL)
    if identity != expected:
        raise ValueError(
            "unsupported benchmark report identity: "
            f"schema_version={identity[0]!r}, artifact_type={identity[1]!r}, "
            f"tool={identity[2]!r}; expected {expected!r}"
        )
    return report


def report_rows(report_path: Path, case: BenchCase) -> list[dict[str, Any]]:
    report = load_bench_report(report_path)
    config = report.get("config", {})
    load = report.get("load", {})
    memory = report.get("memory", {})
    weights_memory = memory.get("weights", {})
    sequence_memory = memory.get("sequence", {})
    workspace_memory = memory.get("workspace", {})
    rows = []
    for test in report.get("tests", []):
        speculative = test.get("speculative", {})
        row = {
            "suite": case.suite,
            "case": case.name,
            "report": str(report_path),
            "label": test.get("label"),
            "kind": test.get("kind"),
            "n_prompt": test.get("n_prompt"),
            "n_gen": test.get("n_gen"),
            "requested_output_tokens": test.get("requested_output_tokens"),
            "target": load.get("target"),
            "artifact_path": report.get("artifact", {}).get("path"),
            "max_context": config.get("max_context"),
            "prefill_chunk": config.get("prefill_chunk"),
            "kv_cache": config.get("kv_cache"),
            "mtp_draft_tokens": config.get("mtp_draft_tokens"),
            "proposal_head": config.get("proposal_head"),
            "decode_path": config.get("decode_path"),
            "decode_graph_primed": config.get("decode_graph_prime", {}).get("primed"),
            "decode_graph_prime_output_tokens": config.get("decode_graph_prime", {}).get(
                "output_tokens"
            ),
            "repetitions": config.get("repetitions"),
            "warmup": config.get("warmup"),
            "load_seconds": load.get("load_seconds"),
            "upload_seconds": load.get("upload_seconds"),
            "artifact_bytes_read": load.get("artifact_bytes_read"),
            "host_to_device_bytes": load.get("host_to_device_bytes"),
            "peak_staging_bytes": load.get("peak_staging_bytes"),
            "kv_payload_bytes": memory.get("kv_payload_bytes"),
            "weights_capacity_bytes": weights_memory.get("capacity_bytes"),
            "sequence_capacity_bytes": sequence_memory.get("capacity_bytes"),
            "workspace_capacity_bytes": workspace_memory.get("capacity_bytes"),
            "workspace_peak_bytes": test.get("workspace_peak_bytes"),
            "prefill_tok_s_mean": test.get("prefill_tok_s_mean"),
            "prefill_tok_s_stddev": test.get("prefill_tok_s_stddev"),
            "decode_output_tok_s_mean": test.get("decode_output_tok_s_mean"),
            "decode_output_tok_s_stddev": test.get("decode_output_tok_s_stddev"),
            "decode_engine_tok_s_mean": test.get("decode_engine_tok_s_mean"),
            "decode_engine_tok_s_stddev": test.get("decode_engine_tok_s_stddev"),
            "prepare_seconds_mean": test.get("prepare_seconds_mean"),
            "prefill_seconds_mean": test.get("prefill_seconds_mean"),
            "decode_seconds_mean": test.get("decode_seconds_mean"),
            "total_seconds_mean": test.get("total_seconds_mean"),
            "spec_acceptance_rate": speculative.get("acceptance_rate"),
            "spec_acceptance_length": speculative.get("acceptance_length"),
            "spec_rounds": speculative.get("rounds"),
            "spec_drafted_tokens": speculative.get("drafted_tokens"),
            "spec_accepted_tokens": speculative.get("accepted_tokens"),
            "spec_fallback_steps": speculative.get("fallback_steps"),
            "spec_accepted_per_position": json.dumps(
                speculative.get("accepted_per_position", []), separators=(",", ":")
            ),
            "gpu_name": report.get("environment", {}).get("gpu_name"),
        }
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        (out_dir / "summary.csv").write_text("", encoding="utf-8")
        (out_dir / "summary.json").write_text("[]\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def run_command(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return process.returncode


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    cases: Sequence[BenchCase],
    commands: Sequence[dict[str, Any]],
) -> None:
    manifest = {
        "artifact_type": "ninfer_bench_matrix_run",
        "schema_version": 2,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "preset": args.preset,
        "primary_mtp_draft_tokens": 3,
        "primary_proposal_head": "optimized",
        "repo_root": str(REPO_ROOT),
        "bench": str(args.bench),
        "artifact": str(args.weights),
        "corpus": str(args.corpus),
        "corpus_tokens": count_corpus_tokens(args.corpus),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "case_count": len(cases),
        "commands": list(commands),
        "notes": [
            "k=3 with the optimized proposal head is the primary MTP path.",
            "Use context_decode and mtp_sweep rows for MTP efficiency decisions.",
            "tg rows use a one-token seed and report G decode tokens after the begin token.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "core", "full"), default="core")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help=".ninfer artifact passed to the bench"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", action="append", default=[], help="suite to run; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N selected cases")
    parser.add_argument("--repetitions", type=int, default=None, help="override all case repetitions")
    parser.add_argument("--warmup", type=int, default=None, help="override all case warmup repetitions")
    parser.add_argument("--dry-run", action="store_true", help="write commands but do not execute")
    parser.add_argument("--resume", action="store_true", help="skip cases with an existing valid JSON report")
    parser.add_argument(
        "--no-build", action="store_true", help="do not build build/bench/ninfer_bench"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.repetitions is not None and args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be nonnegative")

    args.bench = args.bench.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()

    if not args.weights.is_file():
        raise SystemExit(f"weights file not found: {args.weights}")
    corpus_tokens = count_corpus_tokens(args.corpus)

    all_cases = build_cases(args.preset)
    cases = filtered_cases(all_cases, args.suite, args.limit)
    if not cases:
        raise SystemExit("selected matrix is empty")
    max_prompt = max_prompt_in_cases(cases)
    if max_prompt > corpus_tokens:
        raise SystemExit(
            f"selected matrix needs prompt length {max_prompt}, but corpus has {corpus_tokens} tokens"
        )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "profiles/bench" / f"ninfer-{args.preset}-{utc_stamp()}"
    out_dir = out_dir.expanduser().resolve()
    json_dir = out_dir / "json"
    log_dir = out_dir / "logs"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    command_records: list[dict[str, Any]] = []
    commands_sh: list[str] = []
    for case in cases:
        report_path = json_dir / case.suite / f"{case.name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        base_args = [
            str(args.bench),
            "--weights",
            str(args.weights),
            "--corpus",
            str(args.corpus),
            "--device",
            str(args.device),
            *case.args,
            "--output",
            "json",
            "--output-file",
            str(report_path),
        ]
        command = add_repetition_args(base_args, case, args.repetitions, args.warmup)
        command_records.append(
            {
                "suite": case.suite,
                "case": case.name,
                "report": str(report_path),
                "notes": case.notes,
                "command": command,
            }
        )
        commands_sh.append(shell_join(command))
    commands_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands_sh) + "\n"
    (out_dir / "commands.sh").write_text(commands_text, encoding="utf-8")
    write_manifest(out_dir, args, cases, command_records)

    if args.dry_run:
        print(f"wrote dry-run matrix to {out_dir}")
        print(f"cases: {len(cases)}")
        return 0

    failures: list[dict[str, Any]] = []
    if not args.no_build:
        build_stdout = log_dir / "build.stdout.txt"
        build_stderr = log_dir / "build.stderr.txt"
        rc = run_command(
            ["cmake", "--build", "build", "-j", "--target", "ninfer_bench"],
            build_stdout,
            build_stderr,
        )
        if rc != 0:
            failures.append(
                {
                    "case": "build",
                    "returncode": rc,
                    "stdout": str(build_stdout),
                    "stderr": str(build_stderr),
                }
            )
            (out_dir / "failures.json").write_text(
                json.dumps(failures, indent=2) + "\n", encoding="utf-8"
            )
            print(f"build failed; see {build_stderr}", file=sys.stderr)
            return 1

    if not args.bench.is_file():
        raise SystemExit(f"bench binary not found after build: {args.bench}")

    for index, record in enumerate(command_records, start=1):
        case = cases[index - 1]
        report_path = Path(record["report"])
        if args.resume and report_path.is_file():
            try:
                load_bench_report(report_path)
                print(f"[{index}/{len(cases)}] skip {case.name} (existing report)")
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        stdout_path = log_dir / f"{case.suite}.{case.name}.stdout.txt"
        stderr_path = log_dir / f"{case.suite}.{case.name}.stderr.txt"
        print(f"[{index}/{len(cases)}] run {case.suite}/{case.name}")
        rc = run_command(record["command"], stdout_path, stderr_path)
        if rc != 0:
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "returncode": rc,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )
            print(f"  failed with rc={rc}; see {stderr_path}", file=sys.stderr)
            continue
        if not report_path.is_file():
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "returncode": rc,
                    "error": "report file was not created",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )

    rows: list[dict[str, Any]] = []
    for record, case in zip(command_records, cases, strict=True):
        report_path = Path(record["report"])
        if not report_path.is_file():
            continue
        try:
            rows.extend(report_rows(report_path, case))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "report": str(report_path),
                    "error": f"failed to parse report: {exc}",
                }
            )

    write_summary(rows, out_dir)
    if failures:
        (out_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed with {len(failures)} failure(s); see {out_dir / 'failures.json'}")
        return 1

    print(f"completed {len(cases)} cases")
    print(f"summary: {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
