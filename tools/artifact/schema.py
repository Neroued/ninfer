"""V3 directory records and structural validation, using only the standard library."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import prod
from typing import ClassVar, Mapping, Sequence, TypeAlias

from .layouts import align_up, encoded_size, get_layout

U64_MAX = (1 << 64) - 1
RAW_BYTES_V1 = "raw_bytes_v1"
ACTIVATION_POLICIES = frozenset(("A16Only", "AllowA8", "AllowA4"))


class ArtifactError(ValueError):
    """A directory or file does not satisfy the NInfer v3 contract."""


def integer(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= U64_MAX:
        raise ArtifactError(f"{label}: expected integer in [{minimum},{U64_MAX}]")
    return value


def identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ArtifactError(f"{label}: expected nonempty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ArtifactError(f"{label}: invalid UTF-8 string") from error
    return value


def shape(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 16:
        raise ArtifactError(f"{label}: expected shape with rank 0 through 16")
    result = tuple(integer(dim, label, positive=True) for dim in value)
    integer(prod(result), f"{label} element count", positive=True)
    return result


def members(
    value: object,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict:
    if not isinstance(value, dict):
        raise ArtifactError(f"{label}: expected object")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ArtifactError(
            f"{label}: missing {sorted(missing)}, unknown {sorted(extra)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class TensorSpec:
    id: str
    shape: tuple[int, ...]
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    id: str
    bytes: int
    encoding: str = RAW_BYTES_V1


@dataclass(frozen=True, slots=True)
class TensorObject:
    id: str
    shape: tuple[int, ...]
    format: str
    layout: str
    offset: int
    bytes: int
    kind: ClassVar[str] = "tensor"

    def to_json(self) -> dict:
        return {**asdict(self), "shape": list(self.shape), "kind": self.kind}


@dataclass(frozen=True, slots=True)
class ResourceObject:
    id: str
    encoding: str
    offset: int
    bytes: int
    kind: ClassVar[str] = "resource"

    def to_json(self) -> dict:
        return {**asdict(self), "kind": self.kind}


ObjectSpec: TypeAlias = TensorSpec | ResourceSpec
ArtifactObject: TypeAlias = TensorObject | ResourceObject


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str | None
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Directory:
    components: dict
    objects: tuple[ArtifactObject, ...]
    bindings: dict
    uses: tuple[dict, ...]
    files: tuple[FileRecord, ...]
    metadata: dict
    provenance: dict

    @property
    def payload_bytes(self) -> int:
        return sum(file.payload_bytes for file in self.files)

    def to_json(self) -> dict:
        return {
            "components": self.components,
            "objects": [obj.to_json() for obj in self.objects],
            "bindings": self.bindings,
            "uses": list(self.uses),
            "files": [asdict(file) for file in self.files],
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


def validate_encoding(obj: ArtifactObject) -> None:
    """Interpret an object only when a consumer requests its encoding."""
    if isinstance(obj, ResourceObject):
        if obj.encoding != RAW_BYTES_V1:
            raise ArtifactError(
                f"{obj.id}: unsupported resource encoding {obj.encoding!r}"
            )
        return
    try:
        size = encoded_size(obj.layout, obj.format, obj.shape)
        alignment = get_layout(obj.layout).alignment
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{obj.id}: {error}") from error
    integer(size, f"{obj.id} encoded size", positive=True)
    if obj.bytes != size or obj.offset % alignment:
        raise ArtifactError(
            f"{obj.id}: expected {size} encoded bytes and {alignment}-byte alignment, "
            f"got bytes={obj.bytes}, offset={obj.offset}"
        )


def plan_objects(specs: Sequence[ObjectSpec]) -> tuple[ArtifactObject, ...]:
    result = []
    seen = set()
    offset = 0
    for spec in specs:
        name = identifier(spec.id, "object id")
        if name in seen:
            raise ArtifactError(f"duplicate object id {name!r}")
        seen.add(name)
        if isinstance(spec, TensorSpec):
            dims = shape(spec.shape, name)
            try:
                start = align_up(offset, get_layout(spec.layout).alignment)
                size = encoded_size(spec.layout, spec.format, dims)
            except (TypeError, ValueError) as error:
                raise ArtifactError(f"{name}: {error}") from error
            obj = TensorObject(name, dims, spec.format, spec.layout, start, size)
        elif isinstance(spec, ResourceSpec):
            obj = ResourceObject(name, spec.encoding, offset, spec.bytes)
        else:
            raise TypeError(
                f"expected TensorSpec or ResourceSpec, got {type(spec).__name__}"
            )
        integer(obj.bytes, f"{name} bytes", positive=True)
        offset = integer(obj.offset + obj.bytes, f"{name} end")
        validate_encoding(obj)
        result.append(obj)
    if not result:
        raise ArtifactError("objects must not be empty")
    return tuple(result)


def _parse_object(value: object) -> ArtifactObject:
    if not isinstance(value, dict):
        raise ArtifactError("object descriptor must be an object")
    common = {"id", "kind", "offset", "bytes"}
    kind = value.get("kind")
    if kind == "tensor":
        members(value, common | {"shape", "format", "layout"}, set(), "tensor")
    elif kind == "resource":
        members(value, common | {"encoding"}, set(), "resource")
    else:
        raise ArtifactError(f"unsupported object kind {kind!r}")
    name = identifier(value["id"], "object id")
    offset = integer(value["offset"], f"{name} offset")
    size = integer(value["bytes"], f"{name} bytes", positive=True)
    integer(offset + size, f"{name} end")
    if kind == "resource":
        return ResourceObject(name, identifier(value["encoding"], name), offset, size)
    return TensorObject(
        name,
        shape(value["shape"], name),
        identifier(value["format"], name),
        identifier(value["layout"], name),
        offset,
        size,
    )


def binding_parts(
    binding: object,
    objects: Mapping[str, ArtifactObject],
    label: str = "binding",
) -> tuple[tuple[str, int, int], ...]:
    """Resolve flat logical element ranges without interpreting tensor codecs."""
    if not isinstance(binding, dict):
        raise ArtifactError(f"{label}: expected Binding object")

    def tensor(name: object) -> TensorObject:
        name = identifier(name, label)
        obj = objects.get(name)
        if not isinstance(obj, TensorObject):
            raise ArtifactError(f"{label}: {name!r} does not reference a tensor")
        return obj

    if set(binding) == {"object"}:
        obj = tensor(binding["object"])
        return ((obj.id, 0, prod(obj.shape)),)
    members(binding, {"parts"}, set(), label)
    values = binding["parts"]
    if not isinstance(values, list) or not values:
        raise ArtifactError(f"{label}: parts must be a nonempty array")
    result = []
    total = 0
    for part in values:
        members(part, {"object", "range"}, set(), label)
        obj = tensor(part["object"])
        bounds = part["range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ArtifactError(f"{label}: range must contain two integers")
        begin, end = (integer(v, label) for v in bounds)
        if not begin < end <= prod(obj.shape):
            raise ArtifactError(
                f"{label}: range {bounds} is outside {obj.id} {obj.shape}"
            )
        total = integer(total + end - begin, f"{label} element count", positive=True)
        result.append((obj.id, begin, end))
    return tuple(result)


def _files(values: object, entry_name: str | None) -> tuple[FileRecord, ...]:
    if not isinstance(values, list) or not values:
        raise ArtifactError("files must be a nonempty array")
    result = []
    seen = {entry_name} if entry_name is not None else set()
    total = 0
    for index, value in enumerate(values):
        members(value, {"path", "payload_bytes"}, set(), f"files[{index}]")
        name = value["path"]
        if index == 0:
            if name is not None:
                raise ArtifactError("files[0].path must be null")
        else:
            name = identifier(name, f"files[{index}].path")
            if name in (".", "..") or "/" in name or "\\" in name or name in seen:
                raise ArtifactError(
                    f"files[{index}]: invalid or duplicate sibling filename {name!r}"
                )
            seen.add(name)
        size = integer(
            value["payload_bytes"], f"files[{index}] payload bytes", positive=True
        )
        total = integer(total + size, "total payload bytes", positive=True)
        result.append(FileRecord(name, size))
    return tuple(result)


def parse_directory(value: object, *, entry_name: str | None = None) -> Directory:
    root = members(
        value,
        {"components", "objects", "bindings", "uses", "files"},
        {"metadata", "provenance"},
        "directory",
    )
    files = _files(root["files"], entry_name)
    raw_objects = root["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ArtifactError("objects must be a nonempty array")
    objects = tuple(_parse_object(item) for item in raw_objects)
    index = {}
    previous_end = 0
    payload_bytes = sum(file.payload_bytes for file in files)
    for obj in objects:
        if obj.id in index:
            raise ArtifactError(f"duplicate object id {obj.id!r}")
        if obj.offset < previous_end or obj.offset + obj.bytes > payload_bytes:
            raise ArtifactError(
                f"{obj.id}: object ranges overlap, are out of order, or exceed payload"
            )
        previous_end = obj.offset + obj.bytes
        index[obj.id] = obj

    components = root["components"]
    if not isinstance(components, dict) or "text" not in components:
        raise ArtifactError("components must contain text")
    for name, component in components.items():
        identifier(name, "component id")
        members(component, {"config"}, {"target", "resources", "proposal"}, name)
        if not isinstance(component["config"], dict):
            raise ArtifactError(f"{name}: config must be an object")
        if "target" in component:
            target = identifier(component["target"], f"{name} target")
            if target not in components:
                raise ArtifactError(f"{name}: missing target {target!r}")
        resources = component.get("resources", {})
        if not isinstance(resources, dict):
            raise ArtifactError(f"{name}: resources must be an object")
        for role, object_id in resources.items():
            identifier(role, f"{name} resource role")
            object_id = identifier(object_id, f"{name}/{role}")
            if not isinstance(index.get(object_id), ResourceObject):
                raise ArtifactError(
                    f"{name}/{role}: missing resource object {object_id!r}"
                )
        if "proposal" in component:
            if name != "text":
                raise ArtifactError("only text may declare proposal")
            proposal = members(component["proposal"], {"domain"}, {"rows"}, "proposal")
            if proposal["domain"] == "full":
                if "rows" in proposal:
                    raise ArtifactError("full proposal must omit rows")
            elif proposal["domain"] == "indexed":
                integer(proposal.get("rows"), "proposal rows", positive=True)
            else:
                raise ArtifactError("proposal domain must be full or indexed")

    bindings = root["bindings"]
    if not isinstance(bindings, dict):
        raise ArtifactError("bindings must be an object")
    for name, binding in bindings.items():
        identifier(name, "parameter")
        binding_parts(binding, index, name)
    uses = root["uses"]
    if not isinstance(uses, list):
        raise ArtifactError("uses must be an array")
    seen_uses = set()
    for use in uses:
        members(
            use, {"parameter", "input"}, {"activation_policy", "auxiliaries"}, "Use"
        )
        name = identifier(use["parameter"], "Use parameter")
        source = identifier(use["input"], f"{name} Use input")
        if name not in bindings or (name, source) in seen_uses:
            raise ArtifactError(f"{name}@{source}: missing parameter or duplicate Use")
        seen_uses.add((name, source))
        if "activation_policy" in use:
            policy = identifier(use["activation_policy"], f"{name} activation policy")
            if policy not in ACTIVATION_POLICIES:
                raise ArtifactError(f"{name}: unsupported activation policy {policy!r}")
        auxiliaries = use.get("auxiliaries", {})
        if not isinstance(auxiliaries, dict):
            raise ArtifactError(f"{name}: auxiliaries must be an object")
        for role, binding in auxiliaries.items():
            identifier(role, f"{name} auxiliary role")
            binding_parts(binding, index, f"{name}@{source}/{role}")
    metadata = root.get("metadata", {})
    provenance = root.get("provenance", {})
    if not isinstance(metadata, dict) or not isinstance(provenance, dict):
        raise ArtifactError("metadata and provenance must be objects")
    if "name" in metadata:
        identifier(metadata["name"], "metadata.name")
    return Directory(
        components, objects, bindings, tuple(uses), files, metadata, provenance
    )


def _unique_members(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ArtifactError(f"nonfinite JSON number {value}")


def decode_directory(data: bytes, *, entry_name: str | None = None) -> Directory:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_members,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid directory JSON: {error}") from error
    return parse_directory(value, entry_name=entry_name)


def encode_directory(directory: Directory | dict) -> bytes:
    value = directory.to_json() if isinstance(directory, Directory) else directory
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError) as error:
        raise ArtifactError(f"cannot encode directory JSON: {error}") from error
