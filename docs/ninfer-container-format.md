# NInfer Artifact Container Version 1

> Status: accepted on 2026-07-13, pending implementation.
>
> Authority: this document defines the future `.ninfer` version-1 single-file container, its
> 16-byte binary prefix, strict embedded-JSON metadata schema, payload geometry, boundary with
> registered model code, loading requirements, and evolution rules. It does not define any
> model-specific object inventory, physical tensor layout, source-checkpoint conversion recipe,
> model mathematics, kernel implementation, current command-line behavior, or q5090 migration.
>
> The project name and extension come from [`ninfer-naming.md`](ninfer-naming.md). Persistent tensor
> numeric semantics come from [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md). The currently
> implemented `.qus` artifact remains governed only by
> [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md). This document does not make
> `.ninfer` a current runtime input or change the q5090 contract.

## 1. Decision

A `.ninfer` artifact is a single-file deployment container consisting of:

```text
16-byte binary prefix
strict UTF-8 JSON metadata
alignment padding
persistent tensor/resource payloads
```

The binary prefix identifies the framing revision and gives the exact JSON byte length. The JSON
describes which persistent objects are stored and, for every object, its canonical NInfer name,
kind, logical storage description, payload-relative offset, and byte length. The converter generates
this metadata together with the payload.

The JSON is a closed loading schema, not an arbitrary metadata document. It does not contain a model
graph, execution schedule, binding program, kernel choice, hardware profile, conversion recipe, or
provenance report. One registered string `model_id` selects compiled model-specific validation and
binding code.

The governing boundary is:

```text
.ninfer JSON
    describes what persistent objects are stored and where their bytes are

compiled model contract
    describes what those named objects mean and how the model consumes them

conversion specification
    describes how source-checkpoint data becomes those objects
```

Tensor offsets, stored lengths, and physical order therefore come from the artifact rather than a
model-sized C++ constant table. Runtime code retains only intrinsically model-semantic information:
the required binding namespace, shape relationships, accepted storage combinations, logical views,
state transitions, and execution behavior.

JSON is chosen because metadata parsing occurs once during model loading and never enters the
inference hot path. Human readability and simpler converter/runtime maintenance are more valuable
here than saving a few hundred kilobytes with a custom fixed-record catalog. Parse time and peak host
memory remain implementation measurements for real artifacts rather than assumed performance facts.

## 2. Terms and ownership

### 2.1 Artifact instance

An **artifact instance** is one concrete `.ninfer` file. It contains one prefix, one JSON value, and
the complete persistent object payload required by its registered model contract.

Two instances may use the same `model_id` while choosing different physical object order or aligned
offsets. They may also contain different legal weight values produced for the same selected
checkpoint. The runtime still accepts only explicitly implemented object, format, layout, and GPU
consumer combinations.

### 2.2 Model contract

A **model contract** is compiled model-specific code and documentation selected by `model_id`. It
implements one exact selected checkpoint and validates that checkpoint's current NInfer binding
namespace. It owns:

- model mathematics, recurrent and cache-state semantics, and native capabilities;
- the complete required tensor and resource inventory;
- canonical object names or deterministic rules that generate them;
- logical shape requirements and relationships;
- allowed numeric-format, storage-layout, and resource-encoding assignments per role;
- fused-object interpretation, tied bindings, logical views, and fixed model slots;
- construction-time object selection, capability dependencies, and binding safety;
- the runtime schedule and closed support path for each selected GPU.

The model contract may generate repeated-layer names and shapes with loops or templates. It need not
duplicate the JSON as a hand-written flat table.

The model contract does **not** own artifact offsets, stored lengths, object-array indexes, JSON byte
positions, container padding, device addresses, or device-arena offsets.

### 2.3 Storage registries

NInfer maintains closed string registries for:

- exact checkpoint-native model identities;
- persistent tensor numeric formats;
- physical tensor storage layouts;
- required-resource encodings.

The relevant string tuple selects a complete registered storage contract. A tensor numeric format
defines logical scalar words or grouped codes/scales; its layout completes the physical byte
interpretation. The JSON cannot construct a new format or layout by providing bit widths, group
sizes, strides, tile sizes, plane offsets, alignment, or arbitrary parameters.

### 2.4 Conversion specification

A **conversion specification** is the external model-specific agreement used to create an artifact.
It owns:

- source repositories, revisions, files, and source tensor names;
- source-to-NInfer object mapping;
- transpose, reshape, permutation, concatenation, slicing, and derived values;
- per-tensor numeric-format and encoder selection;
- physical layout selection;
- object order and payload placement;
- resource construction and canonicalization;
- quality evidence and converter/model parity verification.

These decisions produce the JSON and payload. Their provenance and rationale are not copied into the
runtime artifact.

## 3. Version-1 binary framing

### 3.1 File image

The file has this top-level order:

```text
FilePrefixV1                         16 bytes
metadata_json                       prefix.json_bytes bytes
metadata_alignment_padding          to the next 4096-byte boundary
object payloads                     explicit JSON offsets, array order
exact end of file
```

### 3.2 `FilePrefixV1`

The prefix occupies exactly 16 bytes at file offset zero.

| Offset | Size | Type | Field | Required value |
|---:|---:|---|---|---|
| 0 | 8 | raw bytes | `magic` | `4e 49 4e 46 45 52 00 01` |
| 8 | 8 | little-endian `u64` | `json_bytes` | `1..67108864` |

The magic is:

```text
ASCII "NINFER" + 0x00 + framing revision 0x01
```

A reader compares all eight bytes exactly. The revision is part of the magic rather than a second
JSON or binary version field.

`json_bytes` counts only the UTF-8 JSON bytes. It excludes the 16-byte prefix and all following
alignment padding. Version 1 limits the JSON to 64 MiB so a loader can bound parsing before learning
the declared model identity. There is no NUL terminator.

The `u64` length keeps the prefix as two eight-byte wire words and uses the same checked integer
domain as file positions. It does not enlarge the v1 limit: values above 64 MiB, including any value
with nonzero high 32 bits, are rejected.

### 3.3 Derived positions

Using checked unsigned arithmetic:

```text
json_offset    = 16
metadata_end   = json_offset + json_bytes
payload_offset = align_up(metadata_end, 4096)
```

The actual regular-file size must be at least `payload_offset`. The half-open range
`[metadata_end, payload_offset)` is container padding. A canonical converter writes zero to every
padding byte. Both runtime and offline verifier check this region, which is at most 4095 bytes,
before payload loading.

The 4096-byte boundary separates variable textual metadata from the large payload region on a page
boundary. It does not require individual objects to use 4096-byte alignment and does not determine
device placement.

Version 1 contains no header size, file size, model ID, flags, checksum, directory offset, reserved
bytes, TLV area, or extension block outside the JSON. The opened file supplies actual total size;
the final object span proves the expected end.

## 4. Strict JSON metadata schema

### 4.1 Root object

The complete JSON value is one object with exactly two members. Concrete tensor and resource
objects use the forms in Sections 4.2 and 4.3.

| Member | JSON type | Requirement |
|---|---|---|
| `model_id` | string | nonempty registered exact-model identifier |
| `objects` | array | `1..262144` ordered tensor/resource objects |

No other root member is permitted. The framing revision already defines the JSON schema version, so
the JSON does not repeat `version`, `schema`, `header`, `payload_offset`, or `file_bytes`.

The object array is in converter-selected physical order. Model binding uses `name`, never array
index. Reordering objects can change I/O placement but cannot change which model role a name denotes.

### 4.2 Tensor object

A tensor entry has exactly these seven members:

```json
{
  "name": "text/layers/0/attn/qkv/weight",
  "kind": "tensor",
  "shape": [6144, 2048],
  "format": "Q4G64_F16S",
  "layout": "layout-name",
  "offset": 0,
  "bytes": 3276800
}
```

The values in examples are illustrative and do not allocate a model or layout identifier.

| Member | JSON type | Requirement |
|---|---|---|
| `name` | string | unique canonical NInfer binding name |
| `kind` | string | exactly `"tensor"` |
| `shape` | array of integers | logical tensor shape, rank `0..16` |
| `format` | string | one registered persistent numeric-format name |
| `layout` | string | one registered physical storage-layout name |
| `offset` | integer | byte offset relative to the derived payload start |
| `bytes` | integer | exact positive stored payload length |

The shape describes the NInfer stored object's logical tensor, not its source-checkpoint shape or a
padded physical extent. Every dimension is positive. An empty shape represents one direct-format
scalar. Grouped quantized formats require rank at least one and use the final logical dimension as
their group axis according to [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md).

`bytes` must equal the registered layout's exact encoded-size result for `(format, shape)`. Physical
padding, planes, code/scale regions, tiles, and swizzles are derived entirely from `layout`.

### 4.3 Resource object

A required-resource entry has exactly these five members:

```json
{
  "name": "resource/tokenizer",
  "kind": "resource",
  "encoding": "resource-encoding",
  "offset": 4194304,
  "bytes": 123456
}
```

| Member | JSON type | Requirement |
|---|---|---|
| `name` | string | unique canonical NInfer resource binding name |
| `kind` | string | exactly `"resource"` |
| `encoding` | string | one registered required-resource encoding |
| `offset` | integer | byte offset relative to the derived payload start |
| `bytes` | integer | exact positive enclosing payload length |

A resource encoding defines a complete canonical byte representation, required file alignment,
bounded decoder, structural limits, and validation rules. Resource-local counts, lengths, and
indexes are allowed only when that encoding defines them. The decoder stays within the enclosing
`bytes` span, consumes the complete span, and rejects an otherwise valid prefix followed by
undefined bytes.

“Required resource” means required to be present in the complete artifact. A resource need not be
parsed for a construction-time capability that does not select it.

### 4.4 Exact member sets

The two `kind` values define a closed tagged union:

- a tensor entry has all and only the seven tensor members;
- a resource entry has all and only the five resource members;
- missing, additional, or kind-inappropriate members are rejected;
- no member is optional and JSON `null` is never valid.

In particular, a tensor cannot carry resource `encoding`, and a resource cannot carry `shape`,
`format`, or `layout`.

### 4.5 Canonical names and registered identifiers

`model_id`, object `name`, `layout`, and resource `encoding` are decoded JSON strings whose values:

- contain 1 through 1024 bytes after decoding, except `model_id`, `layout`, and `encoding`, which are
  limited to 128 bytes;
- use lowercase ASCII only;
- begin with `[a-z0-9]` and otherwise contain only `[a-z0-9._/-]`;
- contain no whitespace, NUL, control character, or Unicode normalization ambiguity;
- are case-sensitive identifiers, not display labels;
- never trigger filesystem, URL, plugin, or dynamic-library lookup.

Object names are unique across tensors and resources in the complete artifact. They are NInfer
binding names, not Hugging Face or other source-checkpoint tensor names.

Tensor `format` is the deliberate exception to the lowercase identifier grammar: it is an exact
case-sensitive canonical name from [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md), such as
`BF16`, `I32`, or `Q4G64_F16S`.

### 4.6 Integer rules

Every `shape` element, `offset`, and `bytes` value must be a JSON number token matching:

```text
0|[1-9][0-9]*
```

Negative signs, decimal points, exponents, leading zeros, quoted numbers, and conversion through an
IEEE-754 `double` are forbidden. The parser converts the decimal token directly to an exact unsigned
integer with checked overflow.

Further rules are:

- shape dimensions and `bytes` are positive;
- every JSON integer and the actual file size are at most `9007199254740991` (`2^53 - 1`), preserving
  exact round trips in common C++, Python, and JavaScript tooling;
- all absolute file positions formed as `payload_offset + offset` are additionally representable by
  the required seekable-file interface;
- all element-count, encoded-size, alignment, and file-position calculations use checked arithmetic;
- the selected model contract may impose much smaller rank, dimension, object-count, resource, and
  total-size limits.

Boolean values are not integers for this schema.

## 5. Registered string identities

### 5.1 Tensor numeric formats

The only tensor `format` strings initially accepted by NInfer are:

```text
BF16
FP32
I32
Q4G64_F16S
Q5G64_F16S
Q6G64_F16S
W8G32_F16S
```

Their logical meanings come only from [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md). These
strings are closed identities, not a grammar. A parser does not construct an unknown format by
splitting a string into bit width, group size, or scale type.

### 5.2 Model, layout, and resource registries

`model_id`, `layout`, and resource `encoding` values are project-global registered strings. Each
value has one immutable meaning and is never reassigned after retirement.

A layout definition owns:

- byte order and bit packing;
- code and scale placement;
- plane, row, expert, tile, and swizzle order;
- physical padding and its required contents;
- persistent file alignment;
- the exact encoded-size function;
- logical decode and validation rules.

A layout name is not a kernel or GPU name. Private implementation, tiling, launch, and dispatch may
change without renaming an otherwise identical persistent layout.

The registry state at this decision is:

| Namespace | Registered strings | Retired tombstones |
|---|---|---|
| tensor numeric format | the seven names in Section 5.1 | none |
| `model_id` | none | none |
| tensor `layout` | none | none |
| resource `encoding` | none | none |

Every future model, layout, or resource-encoding registration updates this ledger and links one
authoritative semantic specification in the same change. Retired identifiers remain as tombstones.

### 5.3 Distinguishable storage evolution

`model_id` identifies exact checkpoint-native semantics, not one storage profile or artifact
instance. Changing object spelling, fusion decomposition, persistent inventory, layout, placement,
converter implementation, device placement, or a conforming kernel does not by itself require a new
model ID. Selecting another checkpoint or changing checkpoint-native mathematics, state semantics,
or native behavior does.

Storage evolution under one `model_id` must remain observable in JSON. The persistent model-role
interpretation of an existing tensor signature `(name, kind, shape, format, layout)` or resource
signature `(name, kind, encoding)` is never silently redefined. If fusion order, axis meaning, role
mapping, or another persistent-to-model interpretation changes, the new artifact must change at
least one observable identity field, such as its canonical name, shape, layout/encoding, or
inventory. Offset and object-array order do not qualify because they carry no model semantics.

## 6. JSON syntax, parsing, and canonical emission

### 6.1 Accepted JSON syntax

The metadata must be one valid UTF-8 JSON text without a byte-order mark. Standard JSON whitespace
and member ordering do not change meaning. The nesting depth is at most four (root object, objects
array, object, and tensor shape array), `objects` contains at most 262144 entries, and no decoded
string token exceeds 1024 bytes. A conforming parser rejects:

- invalid UTF-8, an initial BOM, or an unpaired Unicode surrogate;
- comments, trailing commas, NaN, infinity, or other non-JSON extensions;
- more than one top-level JSON value or non-whitespace bytes after it within `json_bytes`;
- duplicate member names at any object level, after JSON escape decoding;
- unknown, missing, or kind-inappropriate schema members;
- a value whose JSON type differs from the exact schema;
- strings or integers that violate Sections 4.5 and 4.6.

Parser configuration is part of correctness. A library mode that silently keeps the first or last
duplicate key, coerces strings to numbers, stores all numbers as `double`, or ignores unknown members
is not conforming.

Although JSON permits escaped strings, validation occurs on the decoded value. For example,
`"name"` and `"\u006eame"` are the same member name for duplicate detection.

### 6.2 Canonical converter output

The runtime accepts every semantically valid encoding described above. The project converter emits
one deterministic representation for reproducible artifacts:

- UTF-8 without BOM;
- no optional whitespace or trailing newline;
- no escape sequence in any member name or string value; every permitted v1 string character is
  emitted as its literal ASCII byte;
- shortest decimal integer spelling;
- root member order: `model_id`, then `objects`;
- tensor member order: `name`, `kind`, `shape`, `format`, `layout`, `offset`, `bytes`;
- resource member order: `name`, `kind`, `encoding`, `offset`, `bytes`.

JSON object member order is not runtime semantics. These rules define converter output and offline
canonical verification, not model binding.

### 6.3 Cold-path representation

The runtime applies an explicit peak metadata-memory budget before payload-sized allocation. The
budget covers the input buffer, decoded token storage, parsed representation, name index, and parser
bookkeeping; exceeding it is a loading error. Parser strategy and internal data structures are not
wire semantics, but a conforming implementation must preserve exact integer tokens, detect duplicate
keys before either value is lost, and enforce every global and model-specific bound. A DOM is
acceptable only when those properties and the peak budget remain enforced.

JSON is consumed only while constructing the model. Before publication, object names and storage
identities are resolved to compiled model roles and closed registry identities. Prefill, decode, and
CUDA Graph execution do not reparse JSON, look up artifact strings to choose behavior, or access the
artifact pathname. Retaining bounded diagnostic copies outside the inference path is an
implementation choice.

## 7. Payload geometry

### 7.1 Ordered explicit ranges

Objects are strictly ordered by `offset` in the JSON array. Each offset is relative to the derived
`payload_offset`; the converter-provided values are authoritative placement metadata.

First require `actual_file_size >= payload_offset` and define:

```text
payload_bytes = actual_file_size - payload_offset
cursor = 0
```

Each object is then validated with checked arithmetic:

```text
alignment = registered_file_alignment(object)

object.offset >= cursor
object.offset % alignment == 0
object.bytes > 0
object.offset + object.bytes <= payload_bytes

cursor = object.offset + object.bytes
```

Every registered v1 file alignment is a positive power of two no greater than 4096. Because
`payload_offset` is 4096-byte aligned, checking the relative object offset also proves absolute file
alignment. Tensor layouts provide alignment and an exact expected byte count for `(format, shape)`.
Resource encodings provide alignment, structural size rules, and model-specific limits. A storage
contract that fundamentally requires greater file alignment does not fit framing v1.

After the final object:

```text
cursor == payload_bytes
```

The array is nonempty, so this rule defines the complete end of file without a serialized file-size
field. It rejects overlapping ranges and trailing bytes while permitting converter-selected aligned
gaps.

The absolute file span for an object is formed only after validation:

```text
[payload_offset + object.offset,
 payload_offset + object.offset + object.bytes)
```

Container gaps before the first relative offset and between objects contain zero in canonical
converter output. The offline verifier checks them. Runtime loading may skip gap bytes. Padding
inside a tensor payload belongs to its registered layout and is validated where required by that
layout or its consumers.

### 7.2 No overlapping aliases

Version 1 does not express tied weights, aliases, or views through overlapping object ranges. One
stored object has one payload span. Compiled model code may bind multiple logical slots or derive
multiple checked views from one named object.

### 7.3 File placement is not device placement

Tensor payloads are already in their final registered persistent representation. Version 1 provides
no load-time persistent repacking, generic unpack-to-dense fallback, or runtime quantization.

File offsets are never device offsets. After validation, runtime code independently chooses device
allocation and residency, copies selected packed payloads without changing their persistent bytes,
and constructs layout descriptors. Adjacent ranges may share one file read. They may share one H2D
copy only when the independently computed device placement preserves the same relative offsets and
allocates every included gap.

The common format has no component directory and guarantees neither one contiguous file span nor one
H2D operation per model capability. The loader supports per-object transfers and may merge ranges as
an implementation optimization.

## 8. Container and compiled-model boundary

### 8.1 Directory validation and name binding

After generic JSON and range validation, the selected model binder validates the full object array by
canonical name. It must:

- reject missing, duplicate, or unexpected objects;
- validate tensor/resource kind and exact shape relationships;
- validate the format/layout or resource-encoding assignment allowed for every role;
- require the complete native-capability inventory selected by the model contract;
- derive fused ranges, tied bindings, and model-visible views;
- define each role's eventual fixed-slot binding;
- provide construction-time capability and residency planning.

The binder can enumerate layers and semantic roles, generate expected names and shapes, look them up
in a temporary name index, mark every entry consumed, and reject any unconsumed entry. It does not
need another flat copy of the JSON directory.

Object-array order is never a binding key.

### 8.2 Closed runtime support

The runtime support check remains a closed whitelist, conceptually:

```text
(model_id, canonical role, shape, format, layout, actual GPU)
```

It is not the Cartesian product of every model, format, layout, and kernel known to the executable.
Unknown model/layout/encoding strings and unavailable combinations are hard errors. There is no
generic fallback obligation.

Version 1 has no q5090-style target-profile member and no GPU field. Runtime code identifies the
actual selected device and checks support for the selected construction plan. A hardware-specific
persistent representation is identified by its exact layout plus the runtime consumer whitelist,
not a second hardware label in JSON.

### 8.3 Construction planning and payload validation

The compiled model contract maps validated names to fixed model roles and selects the objects needed
by the requested native operation. Selection rules, dependencies, logical views, and runtime slots
come from compiled code, not JSON.

All objects must fit the generic typed index and undergo JSON, geometry, inventory, and
storage-identity/size validation. Before any payload-sized host/device allocation, every object in
the selected materialization closure must also fit its target runtime representation: rank,
dimensions, element counts, byte counts, and view arithmetic are checked without narrowing or
implicit reshape. A scalar, high-rank object, or model-private packed object that does not fit a
common tensor descriptor requires an explicit checked model-private representation when selected.

Consumer/GPU and payload-content validation apply to every selected object and its compiled
dependencies before it becomes reachable from published model state. An unavailable native
operation or consumer fails before payload allocation.

Selected tensor validation establishes layout-internal padding, direct-word, grouped-code, and scale
invariants required by the registered storage and model contracts. It may stream through bounded
host storage or run after upload through a validated device path; the format does not require a
second full-size host copy.

For example, W8 validation concerns decoded logical code `-128`/bit pattern `0x80`, not every
unrelated physical byte equal to `0x80`. Direct BF16/FP32 words, including signed zero, infinity, and
NaN, and every I32 word remain format-valid unless a selected model role imposes a narrower value
constraint.

Selected resource decoders read only their enclosing spans and validate every resource-local length,
index, and count before use. Parsed host resources become owned runtime state; inference does not
lazily reopen the artifact.

## 9. Canonical conversion

A converter selects one registered `model_id` before creating output. It then:

1. produces the complete canonical NInfer tensor/resource inventory;
2. assigns the one accepted numeric format, layout, or resource encoding for every object;
3. produces final persistent payload bytes and exact lengths;
4. chooses an I/O-efficient object order and aligned payload-relative offsets;
5. serializes the strict JSON using Section 6.2;
6. writes `FilePrefixV1`, the exact JSON bytes, and zero metadata padding;
7. writes every payload at absolute file position `payload_offset + object.offset` and zeros
   container gaps;
8. verifies JSON, inventory, ranges, layouts, resources, and all payload contents;
9. verifies converter/reference/model parity;
10. publishes the completed file only after every check succeeds.

Payload-relative offsets keep object placement independent of the final JSON byte length. The
converter serializes JSON, derives `payload_offset`, and applies the stored relative offsets without
a metadata/absolute-offset fixed-point calculation.

The converter is the sole producer of file geometry. Runtime source code does not repeat its offset
or order table.

Source names, mapping, quantization encoder, transformation sequence, source hashes, command line,
and selection rationale remain in the model-specific conversion specification and release evidence.

## 10. Required loading behavior

### 10.1 Prefix and JSON

Before allocating payload-sized host or device storage, a conforming runtime loader:

1. opens one regular, seekable artifact file read-only;
2. obtains a nonnegative actual file size representable by the platform offset API and no greater
   than `2^53 - 1`;
3. requires at least 16 bytes and reads the exact prefix;
4. compares all magic bytes and decodes little-endian `json_bytes`;
5. checks `json_bytes` against `1..67108864` and derives bounded metadata/payload positions;
6. reads exactly `json_bytes` through the same descriptor and checks every metadata-alignment byte
   before `payload_offset` is zero;
7. parses strict JSON with duplicate detection, exact unsigned-integer handling, and the v1
   depth/object/string limits enforced during parsing;
8. resolves `model_id` and applies its tighter object, rank, resource, and total-size limits;
9. validates every object schema, identifier, shape, registered storage name, range, and array order;
10. queries the already selected device identity and required capabilities without allocating
    payload storage;
11. builds a temporary name index and invokes model-specific inventory validation and planning.

Short/interrupted reads and all `u64` to `off_t`/`size_t` conversions are handled explicitly. The
loader does not reopen the pathname between validation and payload reads.

A generic inspection tool may parse the closed JSON and explicit ranges for an unknown `model_id`
under its own bounded resource policy, but it cannot claim the artifact is executable.

### 10.2 Allocation, upload, and publication

Host/device placement is computed only from validated metadata and compiled model requirements. The
runtime checks every allocation, subregion, logical view, and device alignment without narrowing.
Staging and destination storage remain alive until their reads, transfers, and validation complete.
On failure after asynchronous work has been submitted, the loader waits for or otherwise proves the
relevant I/O and device work quiescent before releasing associated host or device storage.

Descriptors, parsed resources, and model slots remain temporary until every selected payload has
been read, validated, uploaded where required, and synchronized. Only then may completed model state
be published. Failure exposes no partial new weight store.

Version 1 defines construction-time selection of the initial resident set. It does not define live
incremental loading, unloading, or replacement of a running model.

## 11. Integrity and immutable-input assumption

Version 1 contains no header CRC, JSON checksum, per-object checksum, whole-file digest, signature,
publisher identity, or required sidecar.

Strict JSON, exact ranges, final-EOF matching, model inventory matching, and storage validation detect
many incomplete or inconsistent artifacts. They do not detect every same-size byte change that
remains legal under those rules, and they do not prove which source checkpoint or converter produced
the payload.

The input file remains immutable during loading. One descriptor prevents accidental path replacement
between phases but does not create an atomic snapshot of a file modified concurrently in place.

A release process may publish a digest or conversion report beside an artifact for distribution
verification. It is not required for runtime loading and is not part of the `.ninfer` ABI. If
in-band integrity later becomes a runtime requirement, it receives a new framing revision with one
precisely defined coverage rule; v1 reserves no dormant checksum field.

## 12. Evolution rules

### 12.1 Registered strings

A registered model, layout, or resource-encoding string has one immutable meaning and is never
reused. Changing a persistent layout's legal bytes, decode, encoded-size rule, or file alignment
requires a new layout string. Reimplementing it or fixing code to enforce an existing documented
rule does not.

A new `model_id`, layout, resource encoding, or tensor numeric format can be registered without a new
framing revision only when it satisfies every existing v1 schema, syntax, grammar, count, rank,
integer, size, alignment, range, and loading bound. Otherwise the required common-contract change
uses a new framing revision.

### 12.2 Framing revision

A new framing revision is required for any change to the v1 accepted-file language or byte
interpretation, including the 16-byte prefix; `json_bytes`; accepted JSON syntax and duplicate-key
rules; root/object schema; identifier grammar and bounds; integer, count, rank, or size limits;
payload-start derivation; payload-relative offset meaning; array ordering; alignment, range, gap, or
final-EOF rules; and common integrity behavior. It uses a different complete magic value.

Section 6.2 canonical emission is producer/verifier policy, not runtime file validity. Changing its
whitespace, member-order, or escaping policy alone does not require a framing revision.

There are no ignored JSON members, extension objects, flags, or reserved fields. Adding a member to
the closed schema is a framing change rather than an optional v1 extension.

The project-owned format has no backward-compatibility obligation. Runtime code may remove an old
framing, model, layout, or resource decoder directly. The future `.ninfer` loader does not read q5090
`.qus` files; migration is a separate conversion and implementation task.

## 13. Explicit exclusions

The version-1 prefix and JSON contain none of the following:

- source tensor names, paths, revisions, hashes, or mapping rules;
- quantization recipes, encoders, calibration, clipping, or quality evidence;
- model architecture/config JSON nested inside the metadata, graphs, operators, or schedules;
- binding-slot tables, logical-view catalogs, fusion records, or alias records;
- arbitrary per-layout parameters, strides, padded shapes, planes, tiles, or group settings;
- component catalogs, residency dependencies, current residency, or load-state records;
- GPU/target/profile IDs, kernel selectors, launch geometry, or device addresses;
- converter versions, commands, timestamps, hostnames, CUDA reports, or benchmark results;
- optional-object flags, alternate payload branches, quality tiers, or fallback representations;
- arbitrary key/value fields, nested extension dictionaries, or vendor-private records;
- runtime KV cache, activations, recurrent state, requests, or generated output;
- compression wrappers, encryption, signatures, checksums, sharding, or multi-file manifests;
- filesystem paths, remote URLs, or required sidecars.

Choosing JSON is an encoding decision, not permission to add these subjects. A fact enters the schema
only when the finished artifact cannot be loaded correctly without carrying it.

## 14. Required implementation evidence

When version 1 is implemented, binary-contract and real-artifact verification must cover at least:

- exact magic comparison and a non-symmetric `json_bytes` vector proving little-endian decoding;
- zero, over-64-MiB, overflowing, truncated, or out-of-file JSON lengths;
- invalid UTF-8/BOM, comments, trailing commas, duplicate decoded keys, multiple JSON values, and a
  non-whitespace suffix after the JSON value inside `json_bytes`;
- runtime acceptance of semantically valid noncanonical whitespace, member order, and escaped
  strings, while the offline canonical verifier rejects those spellings;
- missing, unknown, extra, null, or wrong-typed root/object members;
- integer rejection for negative, fraction, exponent, leading-zero, quoted, boolean, `double`-rounded,
  or overflowing forms;
- invalid identifiers, duplicate names, unknown model/layout/encoding strings, and unknown tensor
  formats;
- empty object arrays, invalid tensor/resource kind schemas, excessive rank, zero dimensions, direct
  rank-zero acceptance, and grouped rank-zero rejection;
- exact tensor layout-derived byte size and relative/absolute alignment equivalence at the 4096-byte
  v1 boundary;
- offset overflow, overlap, out-of-order relative ranges, trailing bytes, and wrong final EOF;
- runtime rejection of nonzero metadata-alignment padding, acceptance of a valid non-minimal aligned
  zero gap, and offline rejection of nonzero inter-object container gaps;
- missing, unexpected, duplicate, or attribute-mismatched model objects;
- binding by name independent of converter-selected object-array order;
- direct logical-word preservation and grouped code/scale/tail validation;
- bounded full-span resource decoding and rejection of invalid lengths, indexes, or undefined suffix;
- wrong selected GPU or unavailable consumer before payload/device allocation;
- construction-plan dependency closure and no reachable binding for an unselected object;
- asynchronous upload lifetime and atomic publication on success or failure;
- deterministic canonical converter JSON;
- a real converter-generated artifact and end-to-end inference for every active model contract.

Tests exercise observable framing, JSON schema, numerical, binding, and loading behavior. Source
scans for private parser calls, enum spellings, or implementation order are not substitutes.

## 15. Consequences

The design accepts a small amount of textual metadata and one bounded JSON parse per model load. It
does not enter inference hot paths; real artifacts must report metadata size, object count, parse
time, and peak host memory rather than assuming the cold-path cost is negligible.

The resulting responsibilities are direct:

```text
converter
    writes strict JSON names, shapes, storage identities, order, offsets, lengths, and payloads

generic container reader
    validates the 16-byte prefix, JSON schema, and file ranges

registered model binder
    validates model semantics and plans fixed-slot bindings by canonical name

runtime and kernels
    choose device placement and execute an explicitly supported path
```

The artifact can be inspected and loaded without compiling its offset table, but it cannot teach an
unknown runtime how to execute a new model. JSON replaces the compile-time address table; it does not
replace compile-time model semantics.
