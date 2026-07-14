# NInfer Artifact Container Version 1

> Status: accepted and implemented by the native Python writer/reader/inspector, converter,
> verifier, target binder, and narrow C++ reader on 2026-07-14. The current C++ Engine has not yet
> cut over from `.qus`.
>
> Authority: this document defines the common `.ninfer` version-1 framing, embedded-JSON object
> directory, payload geometry, registries, and the boundary between the generic reader and a
> registered model binder. It does not define a model inventory, source-checkpoint conversion
> recipe, tensor-layout bytes, model mathematics, kernel, GPU policy, or the current C++ Engine.
>
> Project naming comes from [`ninfer-naming.md`](ninfer-naming.md), numeric formats from
> [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md), and registered storage layouts from
> [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md). The first model-specific contract is
> [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md). The currently implemented C++
> Engine continues to consume q5090 v4.2 `.qus` artifacts under
> [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) until its separate migration.

## 1. Decision

A `.ninfer` artifact is one file:

```text
16-byte binary prefix
UTF-8 JSON object directory
padding to a 4096-byte payload boundary
tensor and required-resource payloads
```

The artifact carries the information that a loader cannot recover from compiled model code:

- exact `model_id`;
- persistent object names and kinds;
- tensor shape, numeric format, and storage layout;
- required-resource encoding;
- payload-relative offset and stored byte length.

The JSON is a closed directory schema. It is not a manifest for arbitrary metadata. In particular,
it contains no source tensor name, conversion recipe, fusion/view table, model graph, execution
schedule, kernel choice, GPU profile, runtime state, or provenance report.

The ownership boundary is:

```text
.ninfer directory
    what persistent objects exist and where their bytes are

registered storage layout
    how one tensor payload maps to persistent logical words

compiled model contract
    what the named objects mean, their logical views, and how they are consumed

conversion specification
    how a source checkpoint is transformed into those objects
```

The directory removes the need to compile tensor names, file offsets, physical order, and stored
lengths into C++. It does not make an unknown model executable.

## 2. Version-1 framing

### 2.1 File prefix

The prefix occupies exactly 16 bytes at file offset zero:

| Offset | Size | Type | Field | Required value |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | `4e 49 4e 46 45 52 00 01` |
| 8 | 8 | little-endian `u64` | `json_bytes` | positive JSON byte length |

The magic is ASCII `NINFER`, one zero byte, and framing revision `1`. A reader compares all eight
bytes exactly.

`json_bytes` counts only the JSON text. It excludes the prefix and following padding. A reader uses
checked integer arithmetic and requires the complete declared JSON range to exist in the file.

### 2.2 Derived ranges

```text
json_offset    = 16
metadata_end   = json_offset + json_bytes
payload_offset = align_up(metadata_end, 4096)
```

The file must extend through `payload_offset`. Bytes in `[metadata_end, payload_offset)` are padding
with no loading semantics. A writer naturally emits them as zero; a reader need not inspect them.

All object offsets in JSON are relative to `payload_offset`. The prefix contains no file size,
flags, checksum, object count, target profile, directory offset, or extension area.

## 3. JSON directory

### 3.1 Root

The complete JSON value is one object with exactly two members:

| Member | Type | Meaning |
|---|---|---|
| `model_id` | nonempty string | exact checkpoint-native model identity |
| `objects` | nonempty array | tensor and required-resource objects in physical-offset order |

No additional root member is valid. The framing revision already determines the directory schema,
so JSON does not repeat a schema or version number.

The common reader exposes `model_id`; the compiled target registry decides whether the current
executable and device can run it. A generic inspector may display a structurally valid artifact for
an unknown model, but cannot declare it executable.

### 3.2 Tensor object

A tensor object has all and only these seven members:

```json
{
  "name": "text/layers/3/attention/query_key",
  "kind": "tensor",
  "shape": [7168, 5120],
  "format": "Q4G64_F16S",
  "layout": "row-split-k128-v1",
  "offset": 12582912,
  "bytes": 19496960
}
```

| Member | Type | Meaning |
|---|---|---|
| `name` | nonempty string | canonical model binding name, unique across all objects |
| `kind` | string | exactly `"tensor"` |
| `shape` | array of positive integers | logical stored tensor shape; `[]` is a scalar |
| `format` | string | registered persistent numeric format |
| `layout` | string | registered persistent tensor layout |
| `offset` | nonnegative integer | byte offset relative to `payload_offset` |
| `bytes` | positive integer | exact stored payload length |

The shape is the NInfer stored shape, not the Hugging Face source shape and not a padded physical
extent. Legal rank and format/layout combinations come from the selected layout. The layout's exact
encoded-size function for `(format, shape)` must equal `bytes`.

### 3.3 Required-resource object

A resource object has all and only these five members:

```json
{
  "name": "frontend/tokenizer.json",
  "kind": "resource",
  "encoding": "raw-bytes-v1",
  "offset": 0,
  "bytes": 19989343
}
```

| Member | Type | Meaning |
|---|---|---|
| `name` | nonempty string | canonical model resource name, unique across all objects |
| `kind` | string | exactly `"resource"` |
| `encoding` | string | registered required-resource encoding |
| `offset` | nonnegative integer | byte offset relative to `payload_offset` |
| `bytes` | positive integer | exact enclosing payload length |

`raw-bytes-v1` makes the complete span the resource value. The common reader does not infer a
filename, tokenizer, template, or processor role from it; the model-specific frontend owns that
mapping.

### 3.4 Closed member sets and values

The root and both object kinds use closed member sets. Missing members, extra members, JSON `null`,
wrong JSON types, and fields from the other object kind are invalid. This rule keeps source recipes
and execution facts out of the common artifact contract.

The implementation uses a standard JSON library, then validates the decoded value against this
schema. Integers must be represented by the library as integers rather than booleans or floating
values. Shape dimensions are positive; offsets are nonnegative; lengths are positive. All sums,
products, alignment operations, and conversions to file-offset or memory-size types use checked
arithmetic.

Object names are case-sensitive binding identities, not display labels or source-checkpoint names.
The model binder compares its exact expected names. Format, layout, and encoding strings are exact
closed-registry identities rather than parseable mini-languages.

JSON whitespace, member order, and ordinary equivalent string escaping have no runtime meaning. The
writer uses compact JSON and a stable field order for readable tooling, but that spelling is not an
artifact identity, validity condition, or reproducibility requirement.

## 4. Registered identities

The version-1 registry initially contains:

| Namespace | Registered identities | Authority |
|---|---|---|
| tensor numeric format | `BF16`, `FP32`, `I32`, `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, `W8G32_F16S` | [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md) |
| `model_id` | `qwen3.6-27b` | [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md) |
| tensor layout | `contiguous-le-v1`, `row-split-k128-v1` | [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md) |
| resource encoding | `raw-bytes-v1` | [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md) |

There are no retired tombstones at this revision.

A registered identity has one documented meaning. The common reader does not synthesize unknown
formats or layouts from names, bit widths, shapes, group sizes, or payload lengths.

The layout identity owns byte order, code/scale placement, physical padding, file alignment, and
encoded size. It is not a GPU or kernel name. The model contract owns which registered combination
is accepted for each role and which actual device implementation can consume it.

`model_id` identifies exact checkpoint-native semantics, not a GPU, converter revision, one
quantization run, physical object order, or artifact instance. Target selection additionally uses
the actual device and the model contract's complete storage signatures; no GPU/profile field is
added to JSON.

## 5. Payload geometry

Let:

```text
payload_bytes = actual_file_size - payload_offset
cursor = 0
```

For each object in array order, derive its registered file alignment and validate with checked
arithmetic:

```text
object.offset >= cursor
object.offset % alignment == 0
object.offset + object.bytes <= payload_bytes

cursor = object.offset + object.bytes
```

For tensors, the selected layout must accept `(format, shape)` and its encoded-size result must equal
`object.bytes`. For resources, the selected encoding supplies alignment and length semantics.

These rules make object spans ordered, aligned, non-overlapping, and inside the file. Gaps before or
between objects and unreferenced trailing file bytes have no loading semantics. A normal writer
starts at the first permitted offset, pads only for alignment, and truncates the file at the final
object end.

An object's absolute file span is:

```text
[payload_offset + object.offset,
 payload_offset + object.offset + object.bytes)
```

The container does not express overlapping aliases or views. Compiled model code may bind multiple
logical roles or checked row views to one stored object.

File placement is not device placement. The artifact contains persistent bytes; a runtime may copy,
map, group, or stream their validated spans according to its target-private memory plan. The JSON
contains no device address or runtime repack instruction.

## 6. Common reader and model binder boundary

### 6.1 Generic reader

The generic reader performs only common file work:

1. read and validate the 16-byte prefix;
2. derive the JSON and payload positions;
3. parse the UTF-8 JSON with a standard library;
4. validate the closed root/tensor/resource schema and integer domains;
5. build a unique `name -> object` index;
6. resolve numeric-format, layout, and encoding identities;
7. validate layout compatibility, encoded sizes, alignment, ordering, overlap, and file bounds;
8. expose object descriptors and payload spans.

It does not know Qwen dimensions, source tensors, layer kinds, component flags, fusion records,
logical model views, schedules, kernels, GPU support, or runtime residency.

The JSON and name index are cold-load data. A model binder resolves them once; inference does not
reparse JSON or perform per-layer/per-token directory lookups.

### 6.2 Registered model binder

The binder selected for an executable target validates:

- exact `model_id`;
- the complete required tensor and resource inventory;
- every canonical name and object kind;
- required shape relationships and exact per-role format/layout/encoding assignments;
- fused row partitions, tied bindings, and all model-visible logical views;
- absence of missing and unexpected objects;
- availability of the exact target consumer on the actual selected device.

The binder owns semantic completeness. It may generate repeated layer names and expected shapes
with model-private loops rather than duplicating a flat JSON table in C++.

### 6.3 Payload-content validation

Persistent numeric and layout contracts still apply to actual bytes. Direct words, grouped codes,
binary16 scales, physical padding, and model-role value restrictions are validated by the layout,
target verifier, or target loading path before a consumer relies on those invariants. The common
directory parser does not duplicate those numeric checks.

How a future Engine allocates, uploads, owns, or publishes a loaded product is outside the container
ABI. A Python reference may keep an mmap open and stream rows; a C++ target may materialize packed
device spans. Both consume the same validated directory and layout bytes.

## 7. Writing an artifact

A writer:

1. obtains the complete model-specific object inventory;
2. obtains already transformed direct words or quantized codes/scales from the conversion recipe;
3. encodes each tensor with its registered layout and each resource with its encoding;
4. computes exact lengths and aligned payload-relative offsets;
5. serializes the closed JSON directory;
6. writes the prefix, JSON, metadata padding, and each payload at its declared offset;
7. truncates the output at the final object end.

Payload-relative offsets avoid any fixed-point dependency on the final JSON length. Physical object
order is a converter decision and is never a model binding key.

Source paths, source tensor names, transforms, quantization assignment rationale, converter
revision, command line, timings, and environment belong to the model-specific conversion
specification or an external descriptive conversion report. A sidecar does not participate in
loading and is not required for artifact validity.

The project does not require fixed source/artifact hashes, a clean worktree, byte-identical output
across reruns or machines, an atomic publication protocol, or special interrupted-write cleanup.
Rerunning the local converter replaces an incomplete output.

## 8. Integrity and provenance boundary

Version 1 requires no checksum, digest, signature, publisher identity, or sidecar. Schema, range,
inventory, layout, and source-value verification catch the classes of mistakes relevant to the
project's own conversion workflow; they are not an identity system for externally supplied files.

A descriptive conversion report remains useful. It may record source path, command, recipe and
converter revisions, source-config summary, environment, object/byte summaries, elapsed time, and
final size. These records explain how an artifact was produced without becoming runtime validity or
strict reproducibility requirements.

## 9. Evolution

Adding a model, layout, encoding, or numeric format updates the corresponding registry and one
authoritative semantic document in the same change.

Changing the byte interpretation or encoded-size rule of an existing layout/encoding requires a new
identity. Changing exact checkpoint-native semantics requires a new `model_id`. Converter
implementation, object order, aligned offsets, artifact values, device placement, and a conforming
kernel may change without renaming the model.

A new framing revision is required when the common accepted-file language changes: prefix, JSON
root/object schema, payload-start calculation, offset meaning, or common range interpretation. It
uses a different complete magic value. There are no ignored extension members or reserved JSON
dictionaries in version 1.

Project-owned formats have no compatibility obligation. A runtime may remove an obsolete framing,
model, layout, or encoding directly. A `.ninfer` reader never treats `.qus` as a fallback or alias.

## 10. Explicit exclusions

The prefix and JSON contain none of the following:

- source tensor names, files, revisions, hashes, or transformation rules;
- quantization encoder recipe, calibration data, clipping policy, or quality evidence;
- model dimensions beyond each stored tensor's logical shape;
- model graph, operators, schedules, layer/module enums, fusion records, or logical-view tables;
- arbitrary strides, plane offsets, padded shapes, group settings, or layout parameters;
- runtime component flags, residency plans, device offsets, or cache/state data;
- GPU/profile identity, kernel selector, launch geometry, or fallback representation;
- converter command, environment, timestamps, benchmarks, or provenance report;
- optional extension maps, vendor fields, remote URLs, or required sidecars.

A field enters the common schema only when the finished artifact cannot be located and decoded
correctly without carrying it.

## 11. Required implementation evidence

The native implementation in `tools/artifact/`, `tools/convert/qwen3_6_27b_rtx5090/`,
`tools/reference/qwen3_6_27b_rtx5090/`, and `src/artifact/` satisfies this layer. These checks remain
the compact conformance set for later changes; they do not imply C++ Engine execution support.

Permanent verification is limited to contracts whose regression would load the wrong bytes:

- one small artifact containing all seven numeric formats and one raw resource;
- exact magic and little-endian `json_bytes`, including truncated/out-of-file metadata;
- closed root/tensor/resource fields and JSON types;
- duplicate object names and unknown format/layout/encoding identities;
- offset alignment, overlap, file bounds, and layout-derived size mismatch;
- exact direct-word and Q4/Q5/Q6/W8 code/scale layout round trips;
- Python writer to Python reader/inspector;
- Python writer to the narrow C++ reader;
- the Qwen3.6-27B binder's complete 1172-object inventory and logical views;
- inspection, source probes, and reference inference on one real converter-generated artifact.

This contract does not require canonical-JSON spelling tests, arbitrary malformed-input matrices,
fuzz/resource-exhaustion campaigns, failure injection, interrupted-publication tests, full-file
hash gates, clean-worktree gates, or exact probabilistic-output comparison.

## 12. Consequence

The reusable container mechanism remains intentionally small:

```text
converter
    writes names, storage identities, offsets, lengths, and payloads

generic reader
    parses prefix/JSON and validates directory geometry

registered model binder
    validates the exact inventory and constructs target-private logical views

reference/runtime
    owns memory policy and executes one explicitly supported model target
```

JSON replaces a compiled file-address table. It does not replace compiled model semantics.
