import json
import re
import subprocess
import sys
from pathlib import Path


SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
}
EXCLUDED_ROOTS = {"third_party"}


def collect_candidates(value: object, candidates: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"file_path", "path"} and isinstance(child, str):
                candidates.add(child)
            collect_candidates(child, candidates)
        return

    if isinstance(value, list):
        for child in value:
            collect_candidates(child, candidates)
        return

    if not isinstance(value, str):
        return

    for match in re.finditer(r"(?m)^\*\*\* (?:Add|Update) File: (.+)$", value):
        candidates.add(match.group(1))
    for match in re.finditer(r"(?m)^\*\*\* Move to: (.+)$", value):
        candidates.add(match.group(1))


def main() -> None:
    event = json.load(sys.stdin)
    cwd = Path(event["cwd"]).resolve()
    root = Path(
        subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve()

    candidates: set[str] = set()
    collect_candidates(event.get("tool_input", {}), candidates)

    files: list[str] = []
    for candidate in sorted(candidates):
        path = Path(candidate)
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()

        if not path.is_relative_to(root) or not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in EXCLUDED_ROOTS:
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        files.append(str(path))

    if files:
        subprocess.run(
            [
                "/usr/bin/clang-format",
                "-i",
                "--style=file",
                "--fallback-style=none",
                *files,
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
