from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .backends.registry import backend_names
from .backends.base import BackendDependencyError
from .config import ConfigError, load_config
from .coordinator import Coordinator, load_resume_config, plan_suite, validate_suite
from .result import load_summary

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PREFLIGHT = 3
EXIT_PARTIAL = 4
EXIT_FAILED = 5
EXIT_CANCELLED = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qus-eval", description="Run repository-local model capability evaluations"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate", help="validate configuration and installed backend dependencies"
    )
    validate.add_argument("--config", required=True)
    validate.add_argument("--suite")

    plan = sub.add_parser(
        "plan", help="show work, prerequisites, and expected sample counts"
    )
    plan.add_argument("--config", required=True)
    plan.add_argument("--suite", required=True)
    plan.add_argument(
        "--check-runtime",
        action="store_true",
        help="also resolve keys and installed packages",
    )

    run = sub.add_parser("run", help="run one configured suite")
    run.add_argument("--config", required=True)
    run.add_argument("--suite", required=True)

    status = sub.add_parser("status", help="show persisted run state")
    status.add_argument("--run", required=True)

    resume = sub.add_parser("resume", help="resume an interrupted compatible run")
    resume.add_argument("--run", required=True)

    summarize = sub.add_parser("summarize", help="show the normalized run summary")
    summarize.add_argument("--run", required=True)

    sub.add_parser("list-backends", help="list registered evaluation backends")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        if args.command == "list-backends":
            for name in backend_names():
                console.print(name)
            return EXIT_OK
        if args.command == "status":
            return _show_json(Path(args.run) / "state.json", console)
        if args.command == "summarize":
            payload = load_summary(Path(args.run))
            console.print_json(data=payload)
            return _status_exit(payload.get("status"))
        if args.command == "resume":
            run_dir = Path(args.run).resolve()
            config, suite = load_resume_config(run_dir)
            return _run(config, suite, console, resume_dir=run_dir)

        config = load_config(args.config)
        if args.command == "validate":
            suites = [args.suite] if args.suite else sorted(config.suites)
            for suite in suites:
                validate_suite(config, suite, for_run=True)
                console.print(f"[green]valid[/green] {suite}")
            return EXIT_OK
        if args.command == "plan":
            plans = plan_suite(config, args.suite, for_run=args.check_runtime)
            _show_plan(plans, console)
            return EXIT_OK
        if args.command == "run":
            return _run(config, args.suite, console)
        raise AssertionError(f"unhandled command: {args.command}")
    except BackendDependencyError as exc:
        _print_stderr(f"[red]dependency error:[/red] {exc}")
        return EXIT_PREFLIGHT
    except ConfigError as exc:
        _print_stderr(f"[red]configuration error:[/red] {exc}")
        return EXIT_CONFIG
    except (ImportError, PackageNotFoundError) as exc:
        _print_stderr(f"[red]dependency error:[/red] {exc}")
        return EXIT_PREFLIGHT
    except FileNotFoundError as exc:
        _print_stderr(f"[red]not found:[/red] {exc}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        _print_stderr("[yellow]cancelled[/yellow]")
        return EXIT_CANCELLED
    except Exception as exc:
        _print_stderr(f"[red]error:[/red] {exc}")
        return EXIT_FAILED


def _run(config, suite: str, console: Console, resume_dir: Path | None = None) -> int:
    cancel = threading.Event()
    old_handler = signal.getsignal(signal.SIGINT)

    def request_cancel(signum, frame):
        del signum, frame
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        _print_stderr(
            "[yellow]cancellation requested; waiting for the active backend to checkpoint[/yellow]"
        )

    signal.signal(signal.SIGINT, request_cancel)
    try:
        coordinator = Coordinator(config, suite, cancel)
        run_dir = coordinator.run(resume_dir=resume_dir)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    console.print(f"run directory: {run_dir}")
    summary = load_summary(run_dir)
    console.print(f"status: {summary['status']}")
    return _status_exit(summary.get("status"))


def _show_plan(plans: list[dict], console: Console) -> None:
    table = Table(title="Evaluation plan")
    for column in (
        "Job",
        "Backend",
        "Dataset",
        "Target",
        "Concurrency",
        "Work",
        "Resume",
        "Requirements",
    ):
        table.add_column(column)
    for plan in plans:
        work = f"{plan['total'] if plan['total'] is not None else '?'} {plan['unit']}"
        table.add_row(
            plan["job_id"],
            plan["backend"],
            plan["dataset"],
            plan["target"] or "-",
            str(plan["requested_concurrency"]),
            work,
            "yes" if plan["supports_resume"] else "no",
            ", ".join(plan["requirements"]) or "-",
        )
        for warning in plan["warnings"]:
            console.print(f"[yellow]{plan['job_id']}:[/yellow] {warning}")
    console.print(table)


def _print_stderr(message: str) -> None:
    Console(stderr=True).print(message)


def _show_json(path: Path, console: Console) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    console.print_json(data=payload)
    return _status_exit(payload.get("status"))


def _status_exit(status: str | None) -> int:
    return {
        "completed": EXIT_OK,
        "partial": EXIT_PARTIAL,
        "failed": EXIT_FAILED,
        "cancelled": EXIT_CANCELLED,
        "running": EXIT_OK,
    }.get(status, EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
