"""Migrate one published NInfer v1 artifact to the v2 identity directory in place."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Mapping, Sequence

from tools.artifact.container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    PREFIX_BYTES,
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ArtifactObject,
    ResourceSpec,
    TensorSpec,
    encode_directory,
    plan_objects,
)
from tools.artifact.layouts import align_up
from tools.convert.qwen3_6_27b import inventory as inventory_27b
from tools.convert.qwen3_6_35b_a3b import inventory as inventory_35b


V1_MAGIC = b"NINFER\x00\x01"
BACKUP_SUFFIX = ".v1-metadata-backup"


class MigrationError(ValueError):
    """The file is not one of the published v1 artifacts or cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class PublishedV1Artifact:
    model_id: str
    weights_id: str
    sha256: str
    directory_sha256: str
    inventory: tuple[object, ...]

    @property
    def identity(self) -> ArtifactIdentity:
        return ArtifactIdentity(self.model_id, self.weights_id)


PUBLISHED_V1_ARTIFACTS = (
    PublishedV1Artifact(
        model_id=inventory_27b.MODEL_ID,
        weights_id=inventory_27b.WEIGHTS_ID,
        sha256="74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127",
        directory_sha256="df352a3e3b8555f15fd00d143be213b9e5ff6fb910a1c2390f0785f9a1e2d386",
        inventory=inventory_27b.OBJECT_SPECS,
    ),
    PublishedV1Artifact(
        model_id=inventory_35b.MODEL_ID,
        weights_id=inventory_35b.WEIGHTS_ID,
        sha256="5194407dd6d3092b8c2f81ce41e014b50ca0d6f1ba4e5d8c1492b8652bfa267f",
        directory_sha256="0155278f7e050a4a5c39a697557ea9e2532d44dfc054fd41b496f8b37966732e",
        inventory=inventory_35b.OBJECT_SPECS,
    ),
)


@dataclass(frozen=True, slots=True)
class V1Directory:
    model_id: str
    raw_objects: tuple[Mapping[str, object], ...]
    encoded: bytes
    payload_offset: int
    file_bytes: int


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MigrationError(f"{field} must be a nonempty string")
    return value


def _require_nonnegative_integer(
    value: object, field: str, *, positive: bool = False
) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise MigrationError(f"{field} must be a {qualifier} integer")
    return value


def _read_v1_directory(path: Path) -> V1Directory:
    file_bytes = path.stat().st_size
    if file_bytes < PREFIX_BYTES:
        raise MigrationError("artifact is shorter than the v1 prefix")
    with path.open("rb") as handle:
        prefix = handle.read(PREFIX_BYTES)
        magic, json_bytes = PREFIX.unpack(prefix)
        if magic != V1_MAGIC:
            raise MigrationError("artifact magic is not NInfer v1")
        if json_bytes == 0:
            raise MigrationError("json_bytes must be positive")
        metadata_end = PREFIX_BYTES + json_bytes
        payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
        if metadata_end > file_bytes or payload_offset > file_bytes:
            raise MigrationError(
                "declared JSON or payload start extends beyond the file"
            )
        encoded = handle.read(json_bytes)
    if len(encoded) != json_bytes:
        raise MigrationError("artifact JSON is truncated")
    try:
        root = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid v1 JSON directory: {exc}") from exc
    if not isinstance(root, dict) or frozenset(root) != frozenset(
        ("model_id", "objects")
    ):
        raise MigrationError(
            "v1 directory root must contain exactly model_id and objects"
        )
    model_id = _require_nonempty_string(root["model_id"], "model_id")
    objects = root["objects"]
    if not isinstance(objects, list) or not objects:
        raise MigrationError("objects must be a nonempty array")
    if not all(isinstance(item, dict) for item in objects):
        raise MigrationError("each v1 object entry must be a JSON object")
    return V1Directory(
        model_id=model_id,
        raw_objects=tuple(objects),
        encoded=encoded,
        payload_offset=payload_offset,
        file_bytes=file_bytes,
    )


def _candidate_for_model(
    model_id: str, candidates: Sequence[PublishedV1Artifact]
) -> PublishedV1Artifact:
    matches = [candidate for candidate in candidates if candidate.model_id == model_id]
    if len(matches) != 1:
        raise MigrationError(
            f"v1 model_id {model_id!r} is not one of the two published artifacts"
        )
    return matches[0]


def _validate_inventory(
    directory: V1Directory, candidate: PublishedV1Artifact
) -> tuple[ArtifactObject, ...]:
    if len(directory.raw_objects) != len(candidate.inventory):
        raise MigrationError(
            f"v1 artifact has {len(directory.raw_objects)} objects; expected "
            f"{len(candidate.inventory)}"
        )

    specs: list[ResourceSpec | TensorSpec] = []
    for position, (raw, expected) in enumerate(
        zip(directory.raw_objects, candidate.inventory)
    ):
        if expected.kind == "resource":
            required_members = frozenset(("name", "kind", "encoding", "offset", "bytes"))
            if frozenset(raw) != required_members:
                raise MigrationError(
                    f"v1 resource at position {position} has missing or extra members"
                )
            payload_bytes = _require_nonnegative_integer(
                raw["bytes"], "resource bytes", positive=True
            )
            specs.append(ResourceSpec(expected.name, expected.encoding, payload_bytes))
        elif expected.kind == "tensor":
            required_members = frozenset(
                ("name", "kind", "shape", "format", "layout", "offset", "bytes")
            )
            if frozenset(raw) != required_members:
                raise MigrationError(
                    f"v1 tensor at position {position} has missing or extra members"
                )
            specs.append(
                TensorSpec(
                    expected.name,
                    expected.shape,
                    expected.format,
                    expected.layout,
                )
            )
        else:
            raise TypeError(f"unsupported inventory kind: {expected.kind!r}")

    try:
        planned = plan_objects(specs)
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        raise MigrationError(f"v1 object inventory is invalid: {exc}") from exc
    for position, (raw, expected) in enumerate(zip(directory.raw_objects, planned)):
        if raw != expected.to_json():
            raise MigrationError(
                f"v1 object at position {position} does not match the published "
                f"{candidate.model_id} inventory"
            )
    payload_bytes = directory.file_bytes - directory.payload_offset
    last = planned[-1]
    if last.offset + last.bytes != payload_bytes:
        raise MigrationError(
            "v1 payload length does not match the complete published inventory"
        )
    return planned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows has no O_DIRECTORY and does not expose a portable directory flush through Python.
        # The metadata file itself is flushed before every call to this helper.
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_backup(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _restore_backup(path: Path, backup: Path) -> None:
    data = backup.read_bytes()
    if len(data) < PREFIX_BYTES or data[:8] != V1_MAGIC:
        raise MigrationError(f"invalid migration recovery file: {backup}")
    _, json_bytes = PREFIX.unpack(data[:PREFIX_BYTES])
    if (
        json_bytes == 0
        or align_up(PREFIX_BYTES + json_bytes, PAYLOAD_ALIGNMENT) != len(data)
    ):
        raise MigrationError(f"invalid migration recovery file: {backup}")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    backup.unlink()
    _fsync_directory(path.parent)


def _remove_backup(backup: Path) -> None:
    backup.unlink()
    _fsync_directory(backup.parent)


def _already_v2(
    path: Path, candidates: Sequence[PublishedV1Artifact]
) -> ArtifactIdentity | None:
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic != MAGIC:
        return None
    try:
        with Artifact.open(path) as artifact:
            identity = artifact.identity
    except ArtifactError as exc:
        raise MigrationError(f"invalid v2 artifact: {exc}") from exc
    known = {candidate.identity for candidate in candidates}
    if identity not in known:
        raise MigrationError(f"v2 artifact identity is not migratable: {identity!r}")
    return identity


def _migrate(
    path: Path, candidates: Sequence[PublishedV1Artifact]
) -> ArtifactIdentity:
    path = path.resolve()
    if not path.is_file():
        raise MigrationError(f"artifact does not exist: {path}")
    backup = path.with_name(path.name + BACKUP_SUFFIX)

    try:
        identity = _already_v2(path, candidates)
    except MigrationError:
        if not backup.exists():
            raise
        _restore_backup(path, backup)
        identity = None
    if identity is not None:
        if backup.exists():
            _remove_backup(backup)
        print(f"already v2: {identity.model_id}/{identity.weights_id}")
        return identity

    if backup.exists():
        _restore_backup(path, backup)

    directory = _read_v1_directory(path)
    candidate = _candidate_for_model(directory.model_id, candidates)
    directory_digest = hashlib.sha256(directory.encoded).hexdigest()
    if directory_digest != candidate.directory_sha256:
        raise MigrationError(
            f"v1 directory SHA-256 is {directory_digest}; expected "
            f"{candidate.directory_sha256}"
        )
    planned = _validate_inventory(directory, candidate)

    print("validating published v1 artifact SHA-256...", flush=True)
    artifact_digest = _sha256_file(path)
    if artifact_digest != candidate.sha256:
        raise MigrationError(
            f"v1 artifact SHA-256 is {artifact_digest}; expected {candidate.sha256}"
        )

    encoded = encode_directory(candidate.identity, planned)
    payload_offset = align_up(PREFIX_BYTES + len(encoded), PAYLOAD_ALIGNMENT)
    if payload_offset != directory.payload_offset:
        raise MigrationError(
            "v2 directory does not preserve the existing payload offset"
        )

    with path.open("rb") as handle:
        old_metadata = handle.read(directory.payload_offset)
    if len(old_metadata) != directory.payload_offset:
        raise MigrationError("failed to read the complete v1 metadata region")
    _write_backup(backup, old_metadata)

    try:
        padding_bytes = directory.payload_offset - PREFIX_BYTES - len(encoded)
        with path.open("r+b") as handle:
            handle.seek(PREFIX_BYTES)
            handle.write(encoded)
            handle.write(bytes(padding_bytes))
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            handle.write(PREFIX.pack(MAGIC, len(encoded)))
            handle.flush()
            os.fsync(handle.fileno())

        with Artifact.open(path) as artifact:
            if artifact.identity != candidate.identity:
                raise MigrationError(
                    f"migrated identity is {artifact.identity!r}; expected "
                    f"{candidate.identity!r}"
                )
            if artifact.objects != planned:
                raise MigrationError("migrated object directory changed")
            if artifact.payload_offset != directory.payload_offset:
                raise MigrationError("migrated payload offset changed")
            if artifact.file_bytes != directory.file_bytes:
                raise MigrationError("migrated file length changed")
    except BaseException:
        _restore_backup(path, backup)
        raise

    _remove_backup(backup)
    print(
        f"migrated in place: {candidate.model_id}/{candidate.weights_id} "
        f"({path})"
    )
    return candidate.identity


def migrate(path: str | Path) -> ArtifactIdentity:
    """Migrate one of the two published v1 artifacts in place."""

    return _migrate(Path(path), PUBLISHED_V1_ARTIFACTS)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args(argv)
    try:
        migrate(arguments.artifact)
    except (MigrationError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
