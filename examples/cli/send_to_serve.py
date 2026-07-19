#!/usr/bin/env python3
"""Send one NInfer CLI messages file to an already-running ninfer-serve instance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_cli_messages(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"failed to read messages file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"failed to parse messages file {path}: {error}") from error

    tools = None
    if isinstance(root, dict):
        unknown = set(root) - {"messages", "tools"}
        if unknown:
            raise RuntimeError(
                "messages object contains unsupported fields: " + ", ".join(sorted(unknown))
            )
        tools = root.get("tools")
        root = root.get("messages")
        if tools is not None and not isinstance(tools, list):
            raise RuntimeError("messages object tools must be an array")
    if not isinstance(root, list) or not root or not all(isinstance(item, dict) for item in root):
        raise RuntimeError("messages JSON must be a non-empty message array")
    return root, tools


def endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", type=Path, help="NInfer CLI --messages JSON file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True, help="public model ID configured by the server")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true")
    thinking.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=86400.0)
    parser.add_argument(
        "--api-key-env",
        help="optional environment variable containing a bearer token",
    )
    parser.add_argument("--output", type=Path, help="optional response JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="print the request without sending it")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        messages, tools = load_cli_messages(args.messages)
    except RuntimeError as error:
        print(f"send_to_serve: {error}", file=sys.stderr)
        return 2

    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_completion_tokens": args.max_tokens,
        "seed": args.seed,
        "stream": False,
    }
    if tools is not None:
        body["tools"] = tools
    if args.thinking:
        body["enable_thinking"] = True
    elif args.no_thinking:
        body["enable_thinking"] = False

    request_text = json.dumps(body, ensure_ascii=False, indent=2) + "\n"
    if args.dry_run:
        sys.stdout.write(request_text)
        return 0

    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(
                f"send_to_serve: environment variable {args.api_key_env!r} is not set",
                file=sys.stderr,
            )
            return 2
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint_url(args.base_url),
        data=request_text.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"send_to_serve: HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"send_to_serve: request failed: {error.reason}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(response_bytes)
    except json.JSONDecodeError as error:
        print(f"send_to_serve: server returned invalid JSON: {error}", file=sys.stderr)
        return 1
    output_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
