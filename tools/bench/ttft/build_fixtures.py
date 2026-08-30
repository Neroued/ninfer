#!/usr/bin/env python3
"""Build the frozen text and image inputs used by the Serve TTFT benchmark.

This is an offline maintainer tool.  The benchmark runner only reads its committed output and
never loads a tokenizer, creates images, or calls a token-count endpoint in the timed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "bench" / "fixtures" / "ttft"
TEXT_ROOT = FIXTURE_ROOT / "text"
MEDIA_ROOT = FIXTURE_ROOT / "media"
DEFAULT_TOKENIZER = Path(
    "/home/neroued/models/llm/qwen/Qwen3.8-27B/base-hf-bf16"
)
IMAGE_SIZE = 1024
IMAGE_VISION_TOKENS = 1024
MANY_IMAGE_COUNT = 28
GENERATED_IMAGE_COUNT = 56
VISION_TOKEN_LIMIT = 32_768


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True, use_fast=True
    )


def _template_ids(tokenizer: Any, messages: Sequence[dict[str, Any]], *, tools: Any = None) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    encoded = tokenizer.apply_chat_template(list(messages), **kwargs)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return [int(value) for value in encoded]


def _prompt_tokens(tokenizer: Any, messages: Sequence[dict[str, Any]], *, tools: Any = None) -> int:
    return len(_template_ids(tokenizer, messages, tools=tools))


def _source_ids(tokenizer: Any) -> list[int]:
    messages = json.loads(
        (REPO_ROOT / "examples/cli/messages/long_niah_64k.json").read_text(encoding="utf-8")
    )
    source = messages[-1]["content"]
    return list(tokenizer.encode(source, add_special_tokens=False))


def _fit_user_prompt(
    tokenizer: Any,
    source_ids: Sequence[int],
    target: int,
    *,
    leading_messages: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Decode a meaningful token prefix whose complete chat prompt is exactly target tokens."""

    def candidate(take: int) -> tuple[list[dict[str, str]], int]:
        text = tokenizer.decode(
            list(source_ids[:take]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        messages = [*leading_messages, {"role": "user", "content": text}]
        return messages, _prompt_tokens(tokenizer, messages)

    low, high = 0, min(len(source_ids), target + 256)
    best_take = 0
    while low <= high:
        middle = (low + high) // 2
        _, count = candidate(middle)
        if count < target:
            best_take = middle
            low = middle + 1
        elif count > target:
            high = middle - 1
        else:
            return candidate(middle)[0]

    for take in range(max(0, best_take - 64), min(len(source_ids), best_take + 128) + 1):
        messages, count = candidate(take)
        if count == target:
            return messages
        if count < target:
            for suffix in (" x", "\nZ", "\n0", "\nA.", "\nTTFT"):
                padded = [
                    *messages[:-1],
                    {"role": "user", "content": messages[-1]["content"] + suffix},
                ]
                if _prompt_tokens(tokenizer, padded) == target:
                    return padded
    raise RuntimeError(f"could not construct an exact {target}-token user prompt")


def _common_rendered_tokens(
    tokenizer: Any,
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
) -> int:
    common = 0
    for left_id, right_id in zip(_template_ids(tokenizer, left), _template_ids(tokenizer, right)):
        if left_id != right_id:
            break
        common += 1
    return common


def _common_prefix_messages(
    tokenizer: Any, source_ids: Sequence[int]
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    common_text = tokenizer.decode(
        list(source_ids[:4300]),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    first = [{"role": "user", "content": common_text + "\nBRANCH-A: answer with A."}]
    second = [{"role": "user", "content": common_text + "\nBRANCH-B: answer with B."}]
    first_ids = _template_ids(tokenizer, first)
    second_ids = _template_ids(tokenizer, second)
    common = 0
    for left, right in zip(first_ids, second_ids):
        if left != right:
            break
        common += 1
    if common < 4096:
        raise RuntimeError(f"unmarked fixtures share only {common} rendered tokens")
    return first, second, common


def _render_template(
    tokenizer: Any, messages: Sequence[dict[str, Any]], *, tools: Any = None
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    if not isinstance(rendered, str):
        raise RuntimeError("tokenizer did not render a string chat prompt")
    return rendered


def _stable_frontier(tokenizer: Any, rendered: str, character_boundary: int) -> int:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    return sum(1 for begin, end in offsets if end <= character_boundary and end > begin)


def _shared_system(
    tokenizer: Any, source_ids: Sequence[int], *, variant: str
) -> tuple[str, int, int]:
    user = {"role": "user", "content": "Use the stable reference and answer with one word."}
    for take in range(4140, 4400):
        body = tokenizer.decode(
            list(source_ids[:take]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        body = "\n".join(line.rstrip() for line in body.splitlines())
        system = f"TTFT-STABLE-{variant}\n" + body
        messages = [{"role": "system", "content": system}, user]
        rendered = _render_template(tokenizer, messages)
        start = rendered.find(system)
        if start < 0:
            raise RuntimeError("rendered prompt lost the shared system text")
        frontier = _stable_frontier(tokenizer, rendered, start + len(system))
        total = _prompt_tokens(tokenizer, messages)
        if frontier > 4096 and frontier % 64 != 0 and total + 31 <= 8192:
            return system, frontier, total
    raise RuntimeError("could not construct a non-page-aligned shared system frontier")


def _client_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for index in range(32):
        tools.append(
            {
                "name": f"workspace_action_{index:02d}",
                "description": (
                    "Inspect or validate one repository path and return structured evidence. "
                    f"Fixed benchmark tool {index:02d}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": ["inspect", "compare", "validate"],
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "recursive": {"type": "boolean"},
                                "patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["recursive", "patterns"],
                        },
                    },
                    "required": ["path", "operation", "options"],
                    "additionalProperties": False,
                },
            }
        )
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _normalized_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": False,
            },
        }
        for tool in tools
    ]


def _tool_frontier(tokenizer: Any, tools: Sequence[dict[str, Any]]) -> tuple[int, int]:
    normalized = _normalized_tools(tools)
    messages = [{"role": "user", "content": "Inspect the repository and answer briefly."}]
    rendered = _render_template(tokenizer, messages, tools=normalized)
    last_json = json.dumps(normalized[-1], ensure_ascii=False, separators=(",", ":"))
    compact_end = rendered.rfind(last_json)
    if compact_end >= 0:
        boundary = compact_end + len(last_json)
    else:
        marker = '"name": "workspace_action_31"'
        marker_at = rendered.rfind(marker)
        tools_end = rendered.find("\n</tools>", marker_at)
        if marker_at < 0 or tools_end < 0:
            raise RuntimeError("could not locate the final tool boundary in rendered prompt")
        boundary = tools_end
    frontier = _stable_frontier(tokenizer, rendered, boundary)
    total = _prompt_tokens(tokenizer, messages, tools=normalized)
    if not 2048 <= frontier <= 6144:
        raise RuntimeError(f"tool frontier {frontier} is outside [2048,6144]")
    if total + 31 > 8192:
        raise RuntimeError(f"tool prompt {total} does not fit the 8192-token profile")
    return frontier, total


def _draw_image(index: int) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        ((31 * index + 23) % 256, (67 * index + 41) % 256, (97 * index + 59) % 256),
    )
    draw = ImageDraw.Draw(image)
    for band in range(16):
        x0 = band * 64
        color = (
            (index * 37 + band * 17) % 256,
            (index * 53 + band * 29) % 256,
            (index * 71 + band * 43) % 256,
        )
        draw.rectangle((x0, 0, x0 + 31, IMAGE_SIZE), fill=color)
    for ring in range(10):
        margin = 32 + ring * 42
        color = (
            (index * 11 + ring * 47) % 256,
            (index * 19 + ring * 61) % 256,
            (index * 23 + ring * 73) % 256,
        )
        draw.ellipse(
            (margin, margin, IMAGE_SIZE - margin, IMAGE_SIZE - margin),
            outline=color,
            width=12,
        )
    draw.rectangle((24, 24, 310, 112), fill=(245, 245, 245), outline=(10, 10, 10), width=5)
    draw.text((42, 46), f"NINFER TTFT IMAGE {index:02d}", fill=(5, 5, 5))
    return image


def _write_images() -> list[Path]:
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(GENERATED_IMAGE_COUNT):
        path = MEDIA_ROOT / f"load_{index:02d}.png"
        _draw_image(index).save(path, format="PNG", optimize=True, compress_level=9)
        paths.append(path)
    return paths


def _many_image_messages(paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": _relative(path)} for path in paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _placeholder_messages(paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": "placeholder"} for _ in paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _expanded_media_tokens(tokenizer: Any, paths: Sequence[Path], prompt: str) -> int:
    rendered = _prompt_tokens(tokenizer, _placeholder_messages(paths, prompt))
    return rendered + len(paths) * (IMAGE_VISION_TOKENS - 1)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value), encoding="utf-8")


def _file_record(path: Path, **facts: Any) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), **facts}


def _existing_case(case_name: str, prompt_tokens: int, max_output_tokens: int) -> dict[str, Any]:
    path = REPO_ROOT / "examples" / "cli" / "messages" / f"{case_name}.json"
    return _file_record(
        path,
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
    )


def build(tokenizer_path: Path) -> None:
    tokenizer = _load_tokenizer(tokenizer_path)
    source_ids = _source_ids(tokenizer)
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)

    seed_8k_path = REPO_ROOT / "examples" / "cli" / "messages" / "long_niah_8k.json"
    seed_8k = json.loads(seed_8k_path.read_text(encoding="utf-8"))
    seed_64k_path = REPO_ROOT / "examples" / "cli" / "messages" / "long_niah_64k.json"
    seed_64k = json.loads(seed_64k_path.read_text(encoding="utf-8"))
    independent_8k = _fit_user_prompt(
        tokenizer,
        source_ids,
        7680,
        leading_messages=[
            {
                "role": "system",
                "content": (
                    "TTFT-INDEPENDENT-COLD-LONG-B. "
                    "Answer the supplied document independently."
                ),
            }
        ],
    )
    independent_8k_common = _common_rendered_tokens(tokenizer, seed_8k, independent_8k)
    if independent_8k_common > 3:
        raise RuntimeError(
            "independent 8K fixture does not diverge at the first system-content token"
        )
    independent_8k_path = TEXT_ROOT / "long_8k_independent.json"
    _write_json(independent_8k_path, independent_8k)

    independent_64k = _fit_user_prompt(
        tokenizer,
        source_ids,
        64512,
        leading_messages=[
            {
                "role": "system",
                "content": (
                    "TTFT-INDEPENDENT-COLD-LONG-64K-B. "
                    "Answer the supplied document independently."
                ),
            }
        ],
    )
    independent_64k_common = _common_rendered_tokens(tokenizer, seed_64k, independent_64k)
    if independent_64k_common > 3:
        raise RuntimeError(
            "independent 64K fixture does not diverge at the first system-content token"
        )
    independent_64k_path = TEXT_ROOT / "long_64k_independent.json"
    _write_json(independent_64k_path, independent_64k)

    exact = _fit_user_prompt(tokenizer, source_ids, 8129)
    over = _fit_user_prompt(tokenizer, source_ids, 8193)
    exact_path = TEXT_ROOT / "context_exact.json"
    over_path = TEXT_ROOT / "context_over.json"
    _write_json(exact_path, exact)
    _write_json(over_path, over)

    common_a, common_b, common_tokens = _common_prefix_messages(tokenizer, source_ids)
    common_a_path = TEXT_ROOT / "unmarked_common_a.json"
    common_b_path = TEXT_ROOT / "unmarked_common_b.json"
    _write_json(common_a_path, common_a)
    _write_json(common_b_path, common_b)

    system_a, system_a_frontier, system_a_prompt = _shared_system(
        tokenizer, source_ids, variant="A"
    )
    system_b, system_b_frontier, system_b_prompt = _shared_system(
        tokenizer, source_ids, variant="B"
    )
    system_a_path = TEXT_ROOT / "shared_system_a.txt"
    system_b_path = TEXT_ROOT / "shared_system_b.txt"
    system_a_path.write_text(system_a, encoding="utf-8")
    system_b_path.write_text(system_b, encoding="utf-8")

    tools = _client_tools()
    tool_frontier, tool_prompt = _tool_frontier(tokenizer, tools)
    tools_path = TEXT_ROOT / "client_tools_32.json"
    _write_json(tools_path, tools)

    image_paths = _write_images()
    many_a_paths = image_paths[:MANY_IMAGE_COUNT]
    many_b_paths = image_paths[MANY_IMAGE_COUNT : 2 * MANY_IMAGE_COUNT]
    many_prompt = "Compare all images and answer with the smallest visible image number."
    many_a = _many_image_messages(many_a_paths, many_prompt)
    many_b = _many_image_messages(many_b_paths, many_prompt)
    over_images = _many_image_messages(
        image_paths[:33], "Inspect every image and answer with one short word."
    )
    many_a_path = TEXT_ROOT / "many_image_28_a.json"
    many_b_path = TEXT_ROOT / "many_image_28_b.json"
    over_images_path = TEXT_ROOT / "many_image_33.json"
    _write_json(many_a_path, many_a)
    _write_json(many_b_path, many_b)
    _write_json(over_images_path, over_images)

    many_a_expanded = _expanded_media_tokens(tokenizer, many_a_paths, many_prompt)
    many_b_expanded = _expanded_media_tokens(tokenizer, many_b_paths, many_prompt)
    over_expanded = _expanded_media_tokens(
        tokenizer, image_paths[:33], "Inspect every image and answer with one short word."
    )
    if many_a_expanded + 31 > VISION_TOKEN_LIMIT or many_b_expanded + 31 > VISION_TOKEN_LIMIT:
        raise RuntimeError("28-image fixture exceeds its 32768-token execution envelope")
    if over_expanded <= VISION_TOKEN_LIMIT:
        raise RuntimeError("33-image fixture does not exceed the Vision token envelope")

    encoded_a = sum(path.stat().st_size for path in many_a_paths)
    encoded_b = sum(path.stat().st_size for path in many_b_paths)
    if max(encoded_a, encoded_b) >= 256 * 1024 * 1024:
        raise RuntimeError("28-image fixture exceeds the encoded-media byte limit")

    manifest: dict[str, Any] = {
        "artifact_type": "ninfer_serve_ttft_corpus",
        "schema_version": 1,
        "tokenizer": {
            "identity": "Qwen/Qwen3.6-family",
            "local_qualification_path_name": tokenizer_path.name,
            "enable_thinking": False,
            "add_generation_prompt": True,
        },
        "shapes": {
            "short-32": _existing_case("text_smoke_zh", 30, 32),
            "long-8k-16": _existing_case("long_niah_8k", 7680, 16),
            "long-8k-32": _existing_case("long_niah_8k", 7680, 32),
            "long-8k-independent-32": _file_record(
                independent_8k_path,
                prompt_tokens=7680,
                max_output_tokens=32,
                seed_common_prefix_tokens=independent_8k_common,
            ),
            "long-64k-32": _existing_case("long_niah_64k", 64512, 32),
            "long-64k-independent-32": _file_record(
                independent_64k_path,
                prompt_tokens=64512,
                max_output_tokens=32,
                seed_common_prefix_tokens=independent_64k_common,
            ),
            "long-256k-32": _existing_case("long_niah_256k", 260096, 32),
            "interferer-256": _existing_case("scenario_story_zh_scifi", 127, 256),
            "holder-4096": _existing_case("scenario_story_zh_scifi", 127, 4096),
            "medium-3000": _existing_case("text_smoke_zh", 30, 3000),
            "context-exact": _file_record(
                exact_path, prompt_tokens=8129, max_output_tokens=64
            ),
            "context-over": _file_record(over_path, prompt_tokens=8193),
            "unmarked-common-a": _file_record(
                common_a_path,
                prompt_tokens=_prompt_tokens(tokenizer, common_a),
                common_prefix_tokens=common_tokens,
                max_output_tokens=32,
            ),
            "unmarked-common-b": _file_record(
                common_b_path,
                prompt_tokens=_prompt_tokens(tokenizer, common_b),
                common_prefix_tokens=common_tokens,
                max_output_tokens=32,
            ),
        },
        "shared": {
            "system-a": _file_record(
                system_a_path,
                marked_frontier_tokens=system_a_frontier,
                example_prompt_tokens=system_a_prompt,
            ),
            "system-b": _file_record(
                system_b_path,
                marked_frontier_tokens=system_b_frontier,
                example_prompt_tokens=system_b_prompt,
            ),
            "client-tools-32": _file_record(
                tools_path,
                count=32,
                marked_frontier_tokens=tool_frontier,
                example_prompt_tokens=tool_prompt,
            ),
        },
        "media": {
            "source-files": [
                _file_record(REPO_ROOT / "examples/cli/media" / name)
                for name in (
                    "visual_chart.png",
                    "natural_scene.png",
                    "compare_left.png",
                    "compare_right.png",
                    "temporal_events.mp4",
                )
            ],
            "image-chart": _file_record(
                REPO_ROOT / "examples/cli/messages/image_chart.json",
                prompt_tokens=428,
                media_paths=["examples/cli/media/visual_chart.png"],
            ),
            "image-video": _file_record(
                REPO_ROOT / "examples/cli/messages/mixed_image_video.json",
                prompt_tokens=1575,
                media_paths=[
                    "examples/cli/media/visual_chart.png",
                    "examples/cli/media/temporal_events.mp4",
                ],
            ),
            "many-image-28-a": _file_record(
                many_a_path,
                media_items=MANY_IMAGE_COUNT,
                vision_tokens=MANY_IMAGE_COUNT * IMAGE_VISION_TOKENS,
                expanded_prompt_tokens=many_a_expanded,
                encoded_media_bytes=encoded_a,
                image_paths=[_relative(path) for path in many_a_paths],
            ),
            "many-image-28-b": _file_record(
                many_b_path,
                media_items=MANY_IMAGE_COUNT,
                vision_tokens=MANY_IMAGE_COUNT * IMAGE_VISION_TOKENS,
                expanded_prompt_tokens=many_b_expanded,
                encoded_media_bytes=encoded_b,
                image_paths=[_relative(path) for path in many_b_paths],
            ),
            "many-image-33": _file_record(
                over_images_path,
                media_items=33,
                vision_tokens=33 * IMAGE_VISION_TOKENS,
                expanded_prompt_tokens=over_expanded,
                encoded_media_bytes=sum(path.stat().st_size for path in image_paths[:33]),
                image_paths=[_relative(path) for path in image_paths[:33]],
            ),
            "load-images": [
                _file_record(
                    path,
                    dimensions=[IMAGE_SIZE, IMAGE_SIZE],
                    vision_tokens=IMAGE_VISION_TOKENS,
                    raw_patch_payload_bytes=4096 * 1536 * 2,
                )
                for path in image_paths
            ],
        },
    }
    _write_json(FIXTURE_ROOT / "manifest.json", manifest)


def check() -> None:
    manifest_path = FIXTURE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "ninfer_serve_ttft_corpus":
        raise RuntimeError("TTFT corpus manifest has the wrong artifact_type")

    records: list[dict[str, Any]] = []
    records.extend(manifest["shapes"].values())
    records.extend(manifest["shared"].values())
    records.extend(
        value
        for key, value in manifest["media"].items()
        if key not in {"load-images", "source-files"}
    )
    records.extend(manifest["media"]["source-files"])
    records.extend(manifest["media"]["load-images"])
    for record in records:
        path = REPO_ROOT / record["path"]
        if not path.is_file():
            raise RuntimeError(f"missing TTFT fixture: {record['path']}")
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"TTFT fixture hash mismatch: {record['path']}")

    for name in ("system-a", "system-b"):
        frontier = manifest["shared"][name]["marked_frontier_tokens"]
        if frontier <= 4096 or frontier % 64 == 0:
            raise RuntimeError(f"{name} does not have a qualified non-aligned frontier")
    tools = manifest["shared"]["client-tools-32"]
    if tools["count"] != 32 or not 2048 <= tools["marked_frontier_tokens"] <= 6144:
        raise RuntimeError("client tool fixture is outside its qualification bounds")
    for name in ("many-image-28-a", "many-image-28-b"):
        media = manifest["media"][name]
        if media["media_items"] != 28 or media["vision_tokens"] >= VISION_TOKEN_LIMIT:
            raise RuntimeError(f"{name} is outside the legal Vision envelope")
        if media["expanded_prompt_tokens"] + 31 > VISION_TOKEN_LIMIT:
            raise RuntimeError(f"{name} does not fit the 32768-token context")
        if media["encoded_media_bytes"] >= 256 * 1024 * 1024:
            raise RuntimeError(f"{name} exceeds the media byte envelope")
    if manifest["media"]["many-image-33"]["vision_tokens"] <= VISION_TOKEN_LIMIT:
        raise RuntimeError("33-image rejection fixture does not exceed the Vision envelope")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print("TTFT fixtures OK")
        return 0
    build(args.tokenizer.expanduser().resolve())
    check()
    print(f"wrote TTFT fixtures under {_relative(FIXTURE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
