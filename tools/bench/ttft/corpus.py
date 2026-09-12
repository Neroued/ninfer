"""Frozen corpus access for Serve TTFT cases."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "bench" / "fixtures" / "ttft" / "manifest.json"


class CorpusError(RuntimeError):
    pass


def entitlement(prompt_tokens: int, max_output_tokens: int) -> int:
    if prompt_tokens < 0 or max_output_tokens <= 0:
        raise ValueError("prompt tokens must be nonnegative and max output must be positive")
    growth = max(max_output_tokens - 1, 0)
    return 64 * math.ceil((prompt_tokens + growth) / 64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Corpus:
    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST) -> None:
        self.manifest_path = manifest_path.resolve()
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusError(f"cannot read TTFT corpus manifest: {error}") from error
        if self.manifest.get("artifact_type") != "ninfer_serve_ttft_corpus":
            raise CorpusError("TTFT corpus has the wrong artifact_type")
        if self.manifest.get("schema_version") != 1:
            raise CorpusError("unsupported TTFT corpus schema_version")
        self._data_urls: dict[str, str] = {}
        self.validate()

    def _path(self, record_or_path: dict[str, Any] | str) -> Path:
        relative = record_or_path["path"] if isinstance(record_or_path, dict) else record_or_path
        if not isinstance(relative, str) or not relative:
            raise CorpusError("corpus path is missing")
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise CorpusError(f"corpus path escapes the repository: {relative}") from error
        return path

    def _records(self) -> Iterable[dict[str, Any]]:
        yield from self.manifest["shapes"].values()
        yield from self.manifest["shared"].values()
        media = self.manifest["media"]
        for name, value in media.items():
            if name in {"load-images", "source-files"}:
                yield from value
            else:
                yield value

    def validate(self) -> None:
        for record in self._records():
            path = self._path(record)
            if not path.is_file():
                raise CorpusError(f"missing TTFT corpus file: {record['path']}")
            if _sha256(path) != record.get("sha256"):
                raise CorpusError(f"TTFT corpus hash mismatch: {record['path']}")

        shapes = self.manifest["shapes"]
        expected = {
            "short-32": (30, 32, 64),
            "long-8k-16": (7680, 16, 7744),
            "long-8k-32": (7680, 32, 7744),
            "long-8k-independent-32": (7680, 32, 7744),
            "long-64k-32": (64512, 32, 64576),
            "long-64k-independent-32": (64512, 32, 64576),
            **{
                f"rotation-55k-{index}": (55000, 32, 55040)
                for index in range(6)
            },
            "long-256k-32": (260096, 32, 260160),
            "interferer-256": (127, 256, 384),
            "holder-4096": (127, 4096, 4224),
            "medium-3000": (30, 3000, 3072),
            "context-exact": (8129, 64, 8192),
        }
        for name, (prompt, output, expected_entitlement) in expected.items():
            record = shapes.get(name)
            if not isinstance(record, dict):
                raise CorpusError(f"missing required shape: {name}")
            if record.get("prompt_tokens") != prompt or record.get("max_output_tokens") != output:
                raise CorpusError(f"shape facts changed for {name}")
            if entitlement(prompt, output) != expected_entitlement:
                raise CorpusError(f"shape entitlement changed for {name}")

        independent_shapes = (
            ("long-8k-16", "long-8k-independent-32", "mixed-four"),
            ("long-64k-32", "long-64k-independent-32", "64K Host-swap"),
        )
        for seed_name, independent_name, owner in independent_shapes:
            seed = shapes[seed_name]
            independent = shapes[independent_name]
            if (
                seed["path"] == independent["path"]
                or seed["sha256"] == independent["sha256"]
            ):
                raise CorpusError(f"{owner} long requests are not byte-distinct")
            common = independent.get("seed_common_prefix_tokens")
            if not isinstance(common, int) or common < 0 or common > 3:
                raise CorpusError(
                    f"{owner} long requests do not diverge at the first system-content token"
                )

        if not (
            entitlement(7680, 16) + entitlement(127, 256) <= 8192
            and entitlement(7680, 16) + 2 * entitlement(127, 256) > 8192
        ):
            raise CorpusError("cache-pressure entitlement relation no longer holds")
        long_64k_entitlement = entitlement(64512, 32)
        if not (
            long_64k_entitlement <= 65536
            and 2 * long_64k_entitlement > 65536
        ):
            raise CorpusError("64K Host-swap entitlement relation no longer holds")

        rotation = [shapes[f"rotation-55k-{index}"] for index in range(6)]
        if len({record["path"] for record in rotation}) != 6 or len(
            {record["sha256"] for record in rotation}
        ) != 6:
            raise CorpusError("55K rotation shapes are not byte-distinct")
        if any(record.get("max_peer_common_prefix_tokens", 4) > 3 for record in rotation):
            raise CorpusError("55K rotation shapes do not diverge at the first content token")
        original_labels: list[str] = []
        for record in rotation:
            messages = self._read_json(self._path(record))
            if not isinstance(messages, list) or not messages:
                raise CorpusError("55K rotation fixture has no messages")
            system = messages[0].get("content") if isinstance(messages[0], dict) else None
            original, separator, _ = system.partition(" ") if isinstance(system, str) else ("", "", "")
            if not original or not separator:
                raise CorpusError("55K rotation fixture has no system label")
            original_labels.append(original)
        if len(set(original_labels)) != len(rotation):
            raise CorpusError("55K rotation system labels are not distinct")

        second_labels: list[str] = []
        for record, original in zip(rotation, original_labels, strict=True):
            label = record.get("second_cohort_label")
            if (
                not isinstance(label, str)
                or not label
                or record.get("second_cohort_prompt_tokens") != 55000
            ):
                raise CorpusError("55K second-cohort shape facts are invalid")
            if len(original) != len(label):
                raise CorpusError("55K second-cohort label changes the frozen text shape")
            second_labels.append(label)
        if len(set(second_labels)) != len(rotation) or set(second_labels) & set(original_labels):
            raise CorpusError("55K second-cohort roots are not distinct from the first cohort")
        rotation_entitlement = entitlement(55000, 32)
        if not 4 * rotation_entitlement <= 240000 < 5 * rotation_entitlement:
            raise CorpusError("55K rotation Device-KV pressure relation no longer holds")

        for name in ("system-a", "system-b"):
            frontier = self.manifest["shared"][name]["marked_frontier_tokens"]
            if frontier <= 4096 or frontier % 64 == 0:
                raise CorpusError(f"{name} is not a qualified non-page-aligned prefix")
        tools = self.manifest["shared"]["client-tools-32"]
        if tools.get("count") != 32 or not 2048 <= tools.get("marked_frontier_tokens", 0) <= 6144:
            raise CorpusError("client tool fixture is not qualified")

        for name in ("many-image-28-a", "many-image-28-b"):
            media = self.manifest["media"][name]
            if media.get("media_items") != 28 or media.get("vision_tokens", 0) >= 32768:
                raise CorpusError(f"{name} is outside the legal Vision envelope")
            if media.get("expanded_prompt_tokens", 0) + 31 > 32768:
                raise CorpusError(f"{name} does not fit max_context=32768")
            if media.get("encoded_media_bytes", 0) >= 256 * 1024 * 1024:
                raise CorpusError(f"{name} exceeds the encoded media limit")
        if self.manifest["media"]["many-image-33"].get("vision_tokens", 0) <= 32768:
            raise CorpusError("33-image boundary fixture does not exceed the Vision limit")

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusError(f"cannot read corpus JSON {path}: {error}") from error

    def shape(self, name: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self.manifest["shapes"][name])
        except KeyError as error:
            raise CorpusError(f"unknown TTFT shape: {name}") from error

    def shape_messages(self, name: str) -> list[dict[str, Any]]:
        value = self._read_json(self._path(self.shape(name)))
        if not isinstance(value, list) or not value:
            raise CorpusError(f"shape {name} is not a non-empty messages array")
        return value

    def shared_system(self, name: str) -> str:
        try:
            record = self.manifest["shared"][name]
        except KeyError as error:
            raise CorpusError(f"unknown shared fixture: {name}") from error
        return self._path(record).read_text(encoding="utf-8")

    def client_tools(self, *, changed_first: bool = False) -> list[dict[str, Any]]:
        record = self.manifest["shared"]["client-tools-32"]
        tools = self._read_json(self._path(record))
        if not isinstance(tools, list) or len(tools) != 32:
            raise CorpusError("client tool fixture no longer contains 32 tools")
        if changed_first:
            tools[0]["description"] += " Changed identity for the miss control."
        return tools

    @staticmethod
    def _media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".mp4":
            return "video/mp4"
        raise CorpusError(f"unsupported fixture media extension: {path.suffix}")

    def data_url(self, relative: str) -> str:
        cached = self._data_urls.get(relative)
        if cached is not None:
            return cached
        path = self._path(relative)
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as error:
            raise CorpusError(f"cannot read fixture media {relative}: {error}") from error
        value = f"data:{self._media_type(path)};base64,{encoded}"
        self._data_urls[relative] = value
        return value

    def _inline_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = copy.deepcopy(messages)
        for message in result:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            converted: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    raise CorpusError("media message contains a non-object content part")
                part_type = part.get("type")
                if part_type == "text":
                    converted.append({"type": "text", "text": part.get("text", "")})
                elif part_type in {"image", "image_url"}:
                    source = part.get("image") or part.get("image_url")
                    if isinstance(source, dict):
                        source = source.get("url")
                    if not isinstance(source, str):
                        raise CorpusError("image fixture has no source path")
                    url = source if source.startswith("data:") else self.data_url(source)
                    converted.append({"type": "image_url", "image_url": {"url": url}})
                elif part_type in {"video", "video_url"}:
                    source = part.get("video") or part.get("video_url")
                    if isinstance(source, dict):
                        source = source.get("url")
                    if not isinstance(source, str):
                        raise CorpusError("video fixture has no source path")
                    url = source if source.startswith("data:") else self.data_url(source)
                    converted.append({"type": "video_url", "video_url": {"url": url}})
                else:
                    raise CorpusError(f"unsupported fixture content type: {part_type!r}")
            message["content"] = converted
        return result

    def media_messages(self, name: str) -> list[dict[str, Any]]:
        try:
            record = self.manifest["media"][name]
        except KeyError as error:
            raise CorpusError(f"unknown media fixture: {name}") from error
        messages = self._read_json(self._path(record))
        if not isinstance(messages, list) or not messages:
            raise CorpusError(f"media fixture {name} is not a messages array")
        return self._inline_messages(messages)

    def load_image_messages(self, index: int, *, prompt: str = "Answer with the image number.") -> list[dict[str, Any]]:
        images = self.manifest["media"]["load-images"]
        if index < 0 or index >= len(images):
            raise CorpusError(f"load image index is outside [0,{len(images)})")
        url = self.data_url(images[index]["path"])
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
