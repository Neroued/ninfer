#!/usr/bin/env python3
"""Run a self-contained Serve TTFT campaign with fresh-process isolation."""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.ttft.cases import CASES, CaseDefinition
from tools.bench.ttft.diagnostics import attach_generation_diagnostics
from tools.bench.ttft.profiles import COMMON_ARGS, PROFILE_ARGS
from tools.bench.ttft.render import write_campaign_summary
from tools.bench.ttft.report import ReportError


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "out/qwen3_8_27b_nvfp4.ninfer"
RAM_WEIGHTS_ROOT = Path("/dev/shm/ninfer-artifacts")
RUNNER = REPO_ROOT / "tools/bench/run_serve_ttft.py"
OUTPUT_ROOT = REPO_ROOT / "profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8"
HOST = "127.0.0.1"
PORT = 18080

RESOURCE_CASES = (
    "shared-state-working-set-shift",
    "private-state-working-set-shift",
    "private-only-working-set-shift",
    "shared-state-hot-prefix",
    "resume-after-interference-device",
    "resume-after-interference-state-host",
    "resume-after-interference-kv-host",
    "resume-after-interference-both-host",
    "resume-after-interference-evicted",
    "session-alternating-64k-host-swap",
    "session-rotation-55k-host",
    "session-rotation-55k-two-cohort-stream",
)
CONTRACT_CASES = (
    "session-publication-order",
    "pending-overflow",
    "pending-timeout",
    "context-exact",
    "context-over",
    "vision-disabled",
    "vision-envelope-over",
)
LONG_CONTEXT_CASES = ("cold-long-256k",)
PREPROCESS_CASES = ("many-image-28", "many-image-28-thread-1")
# The two-cohort case already runs the complete ordinary 55K rotation as its first phase.
FULL_EXCLUSIONS = {
    *CONTRACT_CASES,
    *LONG_CONTEXT_CASES,
    "many-image-28-thread-1",
    "session-rotation-55k-host",
}
CAMPAIGNS = {
    "smoke": ("cold-short",),
    "resource": RESOURCE_CASES,
    "preemption": (
        "preemption-replay",
        "preemption-snapshot",
        "preemption-snapshot-mtp",
        "preemption-snapshot-dflash2",
        "shared-growth-recovery",
        "host-history-pressure-cancel",
        "vision-growth-replay",
        "agent-continuation-replay",
        "agent-continuation-snapshot",
    ),
    "load": tuple(name for name in CASES
                  if name.startswith(("resident-", "mixed-arrivals-"))),
    "contract": CONTRACT_CASES,
    "long-context": LONG_CONTEXT_CASES,
    "preprocess": PREPROCESS_CASES,
    "full": tuple(name for name in CASES if name not in FULL_EXCLUSIONS),
}


class CampaignError(RuntimeError):
    pass


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stamp() -> str:
    return dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _ensure_port_free() -> None:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(0.2)
    try:
        if connection.connect_ex((HOST, PORT)) == 0:
            raise CampaignError(
                f"{HOST}:{PORT} is already in use; stop the existing service before the campaign"
            )
    finally:
        connection.close()


def _stage_weights(source: Path) -> tuple[Path, dict[str, Any]]:
    """Materialize the immutable artifact once in tmpfs for all fresh Serve processes."""

    source = source.resolve()
    status = source.stat()
    # The container aligns its payload start, not its total file size.
    if status.st_size <= 0:
        raise CampaignError(f"the artifact is empty: {source}")

    fingerprint = (
        f"{status.st_dev:x}-{status.st_ino:x}-{status.st_size:x}-{status.st_mtime_ns:x}"
    )
    cache_prefix = f"{source.stem}-"
    cached = RAM_WEIGHTS_ROOT / f"{cache_prefix}{fingerprint}.ninfer"
    record: dict[str, Any] = {
        "kind": "tmpfs",
        "source": str(source),
        "path": str(cached),
        "source_device": status.st_dev,
        "source_inode": status.st_ino,
        "source_bytes": status.st_size,
        "source_mtime_ns": status.st_mtime_ns,
    }

    RAM_WEIGHTS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    RAM_WEIGHTS_ROOT.chmod(0o700)
    cache_valid = cached.is_file() and cached.stat().st_size == status.st_size

    # Every file under this private directory is a derived copy owned by this controller. Remove
    # interrupted staging files, obsolete source identities, and incomplete copies.
    for candidate in RAM_WEIGHTS_ROOT.iterdir():
        manager_owned = (
            candidate.name.startswith(cache_prefix)
            or candidate.name.startswith(f".{cache_prefix}")
        )
        if manager_owned and (candidate != cached or not cache_valid):
            candidate.unlink(missing_ok=True)

    if cache_valid:
        record["reused"] = True
        record["stage_seconds"] = 0.0
        print(
            f"artifact cache ready: reuse {cached} ({status.st_size / 1024**3:.2f} GiB)",
            flush=True,
        )
        return cached, record

    staging = RAM_WEIGHTS_ROOT / f".{cached.name}.staging-{os.getpid()}"

    filesystem = os.statvfs(RAM_WEIGHTS_ROOT)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < status.st_size:
        raise CampaignError(
            "insufficient /dev/shm capacity for the selected artifact: "
            f"need {status.st_size / 1024**3:.2f} GiB, "
            f"available {available / 1024**3:.2f} GiB"
        )

    started = time.monotonic()
    print(
        f"artifact cache stage: {source} -> {cached} "
        f"({status.st_size / 1024**3:.2f} GiB, one SSD read)",
        flush=True,
    )
    command = [
        "dd",
        f"if={source}",
        f"of={staging}",
        "bs=16M",
        "iflag=direct",
        "conv=fsync",
        "status=progress",
    ]
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            raise CampaignError(f"artifact tmpfs staging failed with status {completed.returncode}")
        staged_size = staging.stat().st_size
        if staged_size != status.st_size:
            raise CampaignError(
                f"artifact tmpfs staging produced {staged_size} bytes, expected {status.st_size}"
            )
        staging.chmod(0o600)
        os.replace(staging, cached)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - started
    record["reused"] = False
    record["stage_seconds"] = elapsed
    print(f"artifact cache ready in {elapsed:.2f}s: {cached}", flush=True)
    return cached, record


class RunningServe:
    def __init__(self, command: Sequence[str], log_path: Path, startup_timeout: float) -> None:
        self.command = list(command)
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._log: Any = None

    def __enter__(self) -> "RunningServe":
        _ensure_port_free()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=REPO_ROOT,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._wait_ready()
            return self
        except BaseException:
            self.stop()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        if self.process is None:
            raise CampaignError("ninfer-serve was not started")
        started = time.monotonic()
        next_notice = started + 5.0
        deadline = started + self.startup_timeout
        while True:
            returncode = self.process.poll()
            if returncode is not None:
                detail = _tail(self.log_path)
                raise CampaignError(
                    f"ninfer-serve exited during startup with status {returncode}"
                    + (f"\n{detail}" if detail else "")
                )

            connection = http.client.HTTPConnection(HOST, PORT, timeout=1.0)
            try:
                connection.request("GET", "/health", headers={"Connection": "close"})
                response = connection.getresponse()
                body = response.read()
                if response.status == 200:
                    try:
                        value = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        value = None
                    if value == {"status": "ok"}:
                        print(
                            f"  serve ready in {time.monotonic() - started:.2f}s",
                            flush=True,
                        )
                        return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()

            now = time.monotonic()
            if now >= deadline:
                raise CampaignError(
                    f"timed out waiting for ninfer-serve after {self.startup_timeout:.1f}s"
                )
            if now >= next_notice:
                print(f"  loading serve: {now - started:.0f}s", flush=True)
                next_notice = now + 5.0
            time.sleep(0.1)

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None


def _server_command(
    serve: Path,
    weights: Path,
    request_log_jsonl: Path,
    common_args: Sequence[str],
    profile_args: Sequence[str],
) -> list[str]:
    return [
        str(serve),
        str(weights),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--request-log-jsonl",
        str(request_log_jsonl),
        *_without_stats_interval((*common_args, *profile_args)),
        "--log-stats-interval-ms", "1000",
    ]


def _without_stats_interval(arguments: Sequence[str]) -> list[str]:
    """Keep old baseline profiles while applying the campaign's shared observation cadence."""
    result = []
    values = iter(arguments)
    for argument in values:
        if argument == "--log-stats-interval-ms":
            next(values, None)
        elif not argument.startswith("--log-stats-interval-ms="):
            result.append(argument)
    return result


def _load_profiles(path: Path | None) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if path is None:
        return COMMON_ARGS, dict(PROFILE_ARGS)

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read profile config {path}: {error}") from error
    if not isinstance(value, dict) or set(value) - {"common_args", "profiles"}:
        raise CampaignError("profile config must contain profiles and optional common_args")
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise CampaignError("profile config profiles must be a nonempty name-to-argument-list map")

    def arguments(raw: Any, name: str) -> tuple[str, ...]:
        if not isinstance(raw, list) or not all(isinstance(arg, str) for arg in raw):
            raise CampaignError(f"profile config {name} must be a list of argument strings")
        return tuple(raw)

    common = arguments(value.get("common_args", []), "common_args")
    parsed: dict[str, tuple[str, ...]] = {}
    for name, raw in profiles.items():
        if not name:
            raise CampaignError("profile names must not be empty")
        parsed[name] = arguments(raw, f"profiles.{name}")
    return common, parsed


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    result = None
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            result = arguments[index + 1]
        elif argument.startswith(f"{option}="):
            result = argument.partition("=")[2]
    return result


def _runner_command(
    definition: CaseDefinition,
    raw_path: Path,
    request_timeout: float,
    request_log_jsonl: Path,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--request-log-jsonl",
        str(request_log_jsonl),
        "--base-url",
        f"http://{HOST}:{PORT}",
        "--case",
        definition.name,
        "--profile-label",
        definition.profile,
        "--timeout-seconds",
        str(request_timeout),
        "--output",
        str(raw_path),
    ]


def _run_client(command: Sequence[str], progress_path: Path) -> int:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as progress:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        try:
            if process.stderr is None:
                raise CampaignError("failed to capture TTFT runner progress")
            for line in process.stderr:
                progress.write(line)
                progress.flush()
                sys.stderr.write(line)
                sys.stderr.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def _read_run(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read run artifact {path}: {error}") from error
    if not isinstance(value, dict) or value.get("artifact_type") != "ninfer_serve_ttft_run":
        raise CampaignError(f"runner produced an invalid artifact: {path}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Profile config JSON: {"common_args": ["--kv-dtype", "fp8"], '
            '"profiles": {"text-cold-8k": ["--max-context", "8192"]}}. '
            "All argument values are strings. A config replaces the built-in profiles entirely; "
            "omitted common_args means no common arguments. Preserve old commands in this file "
            "when running an earlier Serve binary as a baseline. Relative paths resolve against "
            "the invocation directory. The manifest records effective arguments and commands."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--campaign",
        choices=tuple(CAMPAIGNS),
        help=(
            "smoke=one baseline, resource=pressure and Host-rotation cases, "
            "preemption=restore, mixed scheduling, cancellation and Vision pressure cases, "
            "full=composite performance; contract=HTTP boundaries; long-context=long inputs; preprocess=worker control"
        ),
    )
    selection.add_argument(
        "--case",
        dest="selected_cases",
        choices=tuple(CASES),
        action="append",
        help="run one named case; repeat to form a focused campaign",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--serve", type=Path, required=True, help="Serve executable path")
    parser.add_argument("--artifact", type=Path, default=WEIGHTS, help="source .ninfer artifact path")
    parser.add_argument(
        "--profile-config", type=Path,
        help="JSON file with profiles and optional common_args; see format below",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.startup_timeout_seconds <= 0 or args.request_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    if args.selected_cases and len(set(args.selected_cases)) != len(args.selected_cases):
        parser.error("--case must not repeat the same case")
    args.serve = args.serve.expanduser().resolve()
    args.artifact = args.artifact.expanduser().resolve()
    if args.profile_config is not None:
        args.profile_config = args.profile_config.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for required in (args.serve, args.artifact, RUNNER):
        if not required.is_file():
            raise SystemExit(f"required campaign input is missing: {required}")

    campaign_name = args.campaign or ("focused" if args.selected_cases else "resource")
    case_names = (
        tuple(args.selected_cases)
        if args.selected_cases
        else CAMPAIGNS[args.campaign or "resource"]
    )
    definitions = [CASES[name] for name in case_names]
    try:
        common_args, profiles = _load_profiles(args.profile_config)
    except CampaignError as error:
        raise SystemExit(str(error)) from error
    selected_profiles = sorted({case.profile for case in definitions})
    missing_profiles = sorted(set(selected_profiles) - profiles.keys())
    if missing_profiles:
        raise SystemExit(f"cases have no executable Serve profile: {missing_profiles}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (OUTPUT_ROOT / f"{_stamp()}-{campaign_name}").resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"campaign output directory is not empty: {output_dir}")

    try:
        runtime_weights, artifact_cache = _stage_weights(args.artifact)
    except (OSError, CampaignError) as error:
        raise SystemExit(f"cannot prepare RAM artifact cache: {error}") from error

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(definitions) * args.samples
    plans: list[dict[str, Any]] = []
    for definition in definitions:
        for sample in range(1, args.samples + 1):
            suffix = f"sample-{sample:03d}"
            raw_path = output_dir / "raw" / definition.profile / definition.name / f"{suffix}.json"
            progress_path = (
                output_dir / "progress" / definition.profile / definition.name / f"{suffix}.log"
            )
            serve_log = (
                output_dir / "serve" / definition.profile / definition.name / f"{suffix}.log"
            )
            request_log_jsonl = (
                output_dir
                / "request-log"
                / definition.profile
                / definition.name
                / f"{suffix}.jsonl"
            )
            plans.append(
                {
                    "case": definition.name,
                    "profile": definition.profile,
                    "sample": sample,
                    "raw": str(raw_path),
                    "progress": str(progress_path),
                    "serve_log": str(serve_log),
                    "request_log_jsonl": str(request_log_jsonl),
                    "server_command": _server_command(
                        args.serve, runtime_weights, request_log_jsonl,
                        common_args, profiles[definition.profile],
                    ),
                    "runner_command": _runner_command(
                        definition, raw_path, args.request_timeout_seconds, request_log_jsonl
                    ),
                }
            )

    profile_kv_dtypes = {
        name: _option_value((*common_args, *profiles[name]), "--kv-dtype")
        for name in selected_profiles
    }
    kv_dtypes = set(profile_kv_dtypes.values())
    manifest: dict[str, Any] = {
        "artifact_type": "ninfer_serve_ttft_campaign",
        "schema_version": 2,
        "status": "running",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "campaign": campaign_name,
        "selected_cases": list(case_names),
        "samples": args.samples,
        "case_count": len(definitions),
        "run_count": total,
        "model_profile": (
            "qwen3.8-27b/nvfp4"
            if args.artifact == WEIGHTS.resolve()
            else f"artifact:{args.artifact.stem}"
        ),
        "kv_dtype": next(iter(kv_dtypes)) if len(kv_dtypes) == 1 else None,
        "profile_kv_dtypes": profile_kv_dtypes,
        "serve": str(args.serve),
        "weights_source": str(args.artifact),
        "weights_runtime": str(runtime_weights),
        "profile_config": str(args.profile_config) if args.profile_config is not None else None,
        "common_args": list(common_args),
        "profile_args": {name: list(profiles[name]) for name in selected_profiles},
        "stats_interval_ms": 1000,
        "stats_interval_source": "campaign_override",
        "artifact_cache": artifact_cache,
        "output_dir": str(output_dir),
        "plans": plans,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    failures: list[dict[str, Any]] = []
    valid_runs: list[Path] = []
    print(
        f"TTFT campaign={campaign_name} runs={total} output={output_dir}",
        flush=True,
    )

    try:
        for index, plan in enumerate(plans, start=1):
            definition = CASES[plan["case"]]
            raw_path = Path(plan["raw"])
            progress_path = Path(plan["progress"])
            serve_log = Path(plan["serve_log"])
            request_log_jsonl = Path(plan["request_log_jsonl"])
            request_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            phase = "serve_start"
            print(
                f"[{index}/{total}] {definition.name} "
                f"profile={definition.profile} sample={plan['sample']:03d}",
                flush=True,
            )
            try:
                with RunningServe(
                    plan["server_command"],
                    serve_log,
                    args.startup_timeout_seconds,
                ):
                    phase = "runner"
                    returncode = _run_client(plan["runner_command"], progress_path)
                phase = "artifact"
                run = _read_run(raw_path)
                diagnostic_error = attach_generation_diagnostics(run, request_log_jsonl)
                if diagnostic_error is not None:
                    run["diagnostic_error"] = diagnostic_error
                _write_json(raw_path, run)
                server = run.get("server")
                if isinstance(server, dict) and isinstance(server.get("model"), str):
                    valid_runs.append(raw_path)
                phase = "request_outcome" if run.get("constructed") is True else "construction"
                if returncode != 0 or run.get("constructed") is not True:
                    raise CampaignError(
                        f"runner returned {returncode}, status={run.get('status')!r}"
                    )
                resume = next(
                    (request for request in run["requests"] if request["role"] == "resume"),
                    None,
                )
                detail = ""
                if resume is not None and isinstance(resume.get("ttft_ns"), int):
                    detail = f" resume_ttft={resume['ttft_ns'] / 1e6:.3f}ms"
                print(f"  complete{detail}", flush=True)
            except (CampaignError, OSError) as error:
                failure = {
                    "case": definition.name,
                    "profile": definition.profile,
                    "sample": plan["sample"],
                    "phase": phase,
                    "error": str(error),
                    "raw": str(raw_path),
                    "progress": str(progress_path),
                    "serve_log": str(serve_log),
                    "request_log_jsonl": str(request_log_jsonl),
                }
                failures.append(failure)
                _write_json(output_dir / "failures.json", failures)
                print(f"  failed: {error}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
        manifest["completed_runs"] = len(valid_runs)
        manifest["failure_count"] = len(failures)
        manifest["summary_json"] = str(output_dir / "summary.json")
        manifest["summary_csv"] = str(output_dir / "summary.csv")
        manifest["summary_markdown"] = str(output_dir / "summary.md")
        _write_json(manifest_path, manifest)
        try:
            _, paths = write_campaign_summary(output_dir)
            print(f"partial summary: {paths['markdown']}", file=sys.stderr)
        except ReportError as error:
            manifest["summary_error"] = str(error)
            _write_json(manifest_path, manifest)
            print(f"partial summary failed: {error}", file=sys.stderr)
        print(f"interrupted; partial artifacts: {output_dir}", file=sys.stderr)
        return 130

    manifest["status"] = "failed" if failures else "complete"
    manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
    manifest["completed_runs"] = len(valid_runs)
    manifest["failure_count"] = len(failures)
    manifest["summary_json"] = str(output_dir / "summary.json")
    manifest["summary_csv"] = str(output_dir / "summary.csv")
    manifest["summary_markdown"] = str(output_dir / "summary.md")
    _write_json(manifest_path, manifest)

    try:
        _, paths = write_campaign_summary(output_dir)
    except ReportError as error:
        manifest["summary_error"] = str(error)
        _write_json(manifest_path, manifest)
        print(f"campaign summary failed: {error}", file=sys.stderr)
        return 1

    print(f"summary: {paths['markdown']}", flush=True)
    if failures:
        print(
            f"campaign finished with {len(failures)} failure(s): "
            f"{output_dir / 'failures.json'}",
            file=sys.stderr,
        )
        return 1
    print(f"campaign complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
