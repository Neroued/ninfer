#!/usr/bin/env python3
"""Send one NInfer CLI messages file to an already-running ninfer-serve instance."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest, ServeProtocolError
from tools.streaming_http.client import HttpClientError


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


def _media_source(part: dict[str, Any], kind: str, where: str) -> tuple[str, str | None]:
    direct = part.get(kind)
    if isinstance(direct, str) and direct:
        return direct, None

    field = f"{kind}_url"
    value = part.get(field)
    if isinstance(value, str) and value:
        return value, None
    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str) and url:
            detail = value.get("detail") if kind == "image" else None
            if detail is not None and detail != "auto":
                raise RuntimeError(f"{where} image detail must be omitted or 'auto'")
            return url, detail
    raise RuntimeError(f"{where} has no {kind} source")


def _as_data_uri(
    source: str,
    kind: str,
    explicit_media_type: Any,
    media_root: Path,
    where: str,
) -> str:
    if source.startswith(("http://", "https://", "data:")):
        return source

    raw_path = source[7:] if source.startswith("file://") else source
    path = Path(raw_path)
    if not path.is_absolute():
        path = media_root / path
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"{where} failed to read media {path}: {error}") from error

    media_type = explicit_media_type if isinstance(explicit_media_type, str) else None
    if not media_type:
        media_type, _ = mimetypes.guess_type(path.name)
    if not media_type or not media_type.startswith(f"{kind}/"):
        raise RuntimeError(
            f"{where} cannot infer a {kind} media type for {path}; set media_type explicitly"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _convert_content_part(
    part: Any,
    media_root: Path,
    message_index: int,
    part_index: int,
) -> dict[str, Any]:
    where = f"message {message_index} content part {part_index}"
    if not isinstance(part, dict):
        raise RuntimeError(f"{where} must be an object")

    kind = part.get("type")
    if kind == "text" or "text" in part:
        if not isinstance(part.get("text"), str):
            raise RuntimeError(f"{where} text must be a string")
        return dict(part)

    if kind in {"image", "image_url"} or "image" in part or "image_url" in part:
        source, _ = _media_source(part, "image", where)
        url = _as_data_uri(source, "image", part.get("media_type"), media_root, where)
        result = {
            key: value
            for key, value in part.items()
            if key not in {"type", "image", "image_url", "media_type"}
        }
        result.update({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
        return result

    if kind in {"video", "video_url"} or "video" in part or "video_url" in part:
        source, _ = _media_source(part, "video", where)
        url = _as_data_uri(source, "video", part.get("media_type"), media_root, where)
        result = {
            key: value
            for key, value in part.items()
            if key not in {"type", "video", "video_url", "media_type"}
        }
        result.update({"type": "video_url", "video_url": url})
        return result

    raise RuntimeError(f"{where} has unsupported type: {kind!r}")


def to_openai_messages(
    messages: list[dict[str, Any]], media_root: Path
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            if not content:
                raise RuntimeError(f"message {message_index} content array must not be empty")
            item["content"] = [
                _convert_content_part(part, media_root, message_index, part_index)
                for part_index, part in enumerate(content)
            ]
        converted.append(item)
    return converted


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", type=Path, help="NInfer CLI --messages JSON file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
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
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path.cwd(),
        help="base directory for relative CLI media paths (default: current directory)",
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
        messages = to_openai_messages(messages, args.media_root.resolve())
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

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(
                f"send_to_serve: environment variable {args.api_key_env!r} is not set",
                file=sys.stderr,
            )
            return 2

    request = ProtocolRequest(
        protocol="openai_chat",
        path="/v1/chat/completions",
        payload=body,
        stream=False,
    )
    try:
        client = NInferServeClient(normalize_base_url(args.base_url), args.timeout, api_key)
        result = client.prepare(request).execute()
    except (HttpClientError, ServeProtocolError, OSError) as error:
        print(f"send_to_serve: request failed: {error}", file=sys.stderr)
        return 1

    if result.http.error is not None:
        print(f"send_to_serve: request failed: {result.http.error}", file=sys.stderr)
        return 1
    if result.http.status is None or not 200 <= result.http.status < 300:
        detail = result.error_message or result.http.body.decode("utf-8", errors="replace")
        print(f"send_to_serve: HTTP {result.http.status}: {detail}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(result.http.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
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
