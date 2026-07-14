# NInfer Native Artifact Toolchain Implementation Plan

Status: approved for implementation on 2026-07-14

Completed on 2026-07-14. The native `.ninfer` Python writer/reader/inspector, narrow C++ reader,
Qwen3.6-27B converter/verifier/binder, and complete Text/Vision/MTP/multimodal Python reference were
implemented. The current C++ Engine intentionally remains on `.qus` until its separate multi-target
migration.

## Goal

Build the first complete NInfer-native artifact path around the existing Qwen3.6-27B RTX 5090
target:

```text
Qwen3.6-27B BF16 checkpoint
        -> target-private conversion recipe
        -> complete .ninfer artifact
        -> generic reader + Qwen3.6-27B binder
        -> existing Python reference computation
        -> Text + Vision + complete MTP generation
```

This phase proves that the former q5090 all-in-one ABI can be separated into a small common file
mechanism and a checkpoint-private storage, conversion, binding, and reference implementation. It
does not implement the future C++ target program or claim that the current C++ Engine can load a
`.ninfer` artifact.

## Why this phase exists

The former QUS project and q5090 artifact were one vertically coupled Qwen3.6-27B/RTX 5090 product.
The elevated NInfer project instead selects a small number of exact checkpoints and GPUs and
optimizes each pair independently. The project name is no longer tied to Qwen, but this does not
turn NInfer into a generic model runtime.

The current q5090 v4.2 ABI combines all of the following:

- binary framing;
- fixed Qwen3.6-27B dimensions;
- fixed Text, MTP, Vision, and draft-head module kinds;
- fixed model role and source-kind enums;
- fixed fusion and row-view records;
- quantization assignments and physical layouts;
- frontend resources;
- converter and runtime assumptions.

Adding another exact checkpoint would therefore require either another complete file ABI or more
checkpoint branches in common parser and WeightStore code. `.ninfer` must make only the persistent
object directory common. The model inventory, source mapping, logical views, schedules, and GPU
consumer remain target-private.

## Scope boundary

| Common `.ninfer` mechanism | Qwen3.6-27B target-private implementation |
|---|---|
| prefix, JSON, object directory | complete tensor/resource inventory |
| seven persistent numeric formats | Hugging Face source tensor mapping |
| registered storage-layout codecs | slicing, transpose, concatenation, and derived values |
| mmap payload access | quantization assignment and draft-head recipe |
| Python and C++ generic readers | fused-object logical views |
| generic inspection | Text, MTP, Vision, and state schedules |
| raw resource bytes | frontend filenames and their use |

The common layer must not contain Qwen dimensions, layer kinds, model phases, source tensor names,
module/source-kind enums, a model graph, a kernel selector, or a GPU profile.

## Explicit non-goals

- Do not refactor the current C++ Engine, ModelCard, WeightStore, or CUDA execution path.
- Do not make the current C++ Engine read `.ninfer`.
- Do not implement a future target `Program`, materializer, registry, or generation controller.
- Do not implement Qwen3.6-35B-A3B inference or its artifact inventory.
- Do not add a generic Python model interface, generic WeightProvider, source-checkpoint execution
  backend, dynamic graph, or model-family abstraction.
- Do not change the accepted 27B quantization assignment, group sizes, draft-head selection, or
  effective row-split payload packing.
- Do not implement `.qus -> .ninfer` conversion or a `.qus` fallback in the new reader.
- Do not require old and new artifacts, floating activations, sampled outputs, or complete files to
  be byte-identical.
- Do not add attack/injection models, exhaustive malformed-input matrices, failure-residue
  protocols, fixed hashes, clean-worktree gates, or cross-machine byte reproducibility.

## Initial registrations

### Exact model identity

Register:

```text
qwen3.6-27b
```

The `model_id` denotes exact checkpoint-native semantics. It does not denote RTX 5090, a storage
profile, a quantization revision, or one artifact instance. The source/tool target key is:

```text
qwen3_6_27b_rtx5090
```

New code and artifact identities do not use the legacy `q5090` name.

### Persistent tensor layouts

Register only:

```text
contiguous-le-v1
row-split-k128-v1
```

`contiguous-le-v1` supports `BF16`, `FP32`, and `I32`. It stores little-endian logical words in
C-order without internal padding. Its encoded size is `numel(shape) * word_bytes`.

`row-split-k128-v1` preserves the current effective q5090 row-split bytes while giving them an
independent NInfer identity:

- rank-2 logical shape `[N,K]`;
- `K_pad = align_up(K, 128)`;
- Q4/Q5/Q6 use G64 and W8 uses G32;
- low, optional high, and FP16-scale planes;
- 256-byte plane alignment;
- existing low/high-bit order, signed-code interpretation, and scale order;
- encoded size determined completely by `(format, shape)`.

Layout names describe persistent bytes, so they contain neither `q5090` nor `rtx5090`.

### Required-resource encoding

Register one encoding:

```text
raw-bytes-v1
```

It preserves its payload bytes without interpreting a filename, tokenizer, template, or processor.
The 27B binder owns those meanings.

### Frontend resources

The current canonical Python environment has been checked with an isolated local directory. The
artifact carries these six resources:

```text
frontend/tokenizer.json
frontend/tokenizer_config.json
frontend/chat_template.jinja
frontend/generation_config.json
frontend/preprocessor_config.json
frontend/video_preprocessor_config.json
```

They are sufficient for the current `AutoProcessor`, processor tokenizer, chat template, and
`GenerationConfig` paths. Do not copy redundant `vocab.json`, `merges.txt`, `added_tokens.json`, or
`special_tokens_map.json` into the first artifact.

### Complete 27B inventory

Keep the current physical tensor decomposition:

| Component | Tensor objects |
|---|---:|
| Text | 819 |
| draft lm-head and id map | 2 |
| MTP | 12 |
| Vision | 333 |
| total tensors | 1166 |
| frontend resources | 6 |
| total artifact objects | 1172 |

The tensor-format distribution remains:

| Format | Objects |
|---|---:|
| `BF16` | 582 |
| `FP32` | 96 |
| `I32` | 1 |
| `Q4G64_F16S` | 183 |
| `Q5G64_F16S` | 294 |
| `Q6G64_F16S` | 3 |
| `W8G32_F16S` | 7 |

The layout distribution remains 679 contiguous tensors and 487 row-split tensors.

The artifact is one complete route. Text, MTP, Vision, the full head, the optimized draft head and
ID map, and all frontend resources are mandatory. The new converter has no artifact-profile flags
that omit those components.

Object names use a lowercase NInfer role namespace such as `text/...`, `mtp/...`, and `vision/...`.
They do not reuse Hugging Face source names and do not include `.q4`, `.q5`, or `.w8` suffixes because
the JSON `format` field already carries that fact.

The former 1314 segment records and 130 fusion records are not serialized. The 27B model contract
derives Q/K, gate/V, gate/up, MTP projection, tied-role, and other logical views from the stored
objects.

The GDN convolution object corrects one old metadata accident. Its persistent bytes are tap-major
`[4,10240]`, so the new stored shape is `[4,10240]`; the binder forms the target runtime view. This
does not change its payload values or execution order.

## Source organization

Use the following ownership shape:

```text
tools/
├── artifact/                              generic Python .ninfer mechanism only
│   ├── container.py
│   ├── numeric.py
│   ├── layouts.py
│   └── inspect.py
├── convert/
│   ├── common/
│   │   ├── safetensors.py
│   │   └── quantize.py
│   └── qwen3_6_27b_rtx5090/
│       ├── inventory.py                   emitted storage contract, no source names
│       ├── recipe.py                      source mapping and conversion transforms
│       ├── draft_head.py
│       ├── convert.py
│       └── verify.py
├── reference/
│   └── qwen3_6_27b_rtx5090/
│       ├── bindings.py
│       ├── weights.py
│       ├── model.py
│       ├── text.py
│       ├── mtp.py
│       ├── vision.py
│       ├── multimodal.py
│       ├── sampling.py
│       ├── state.py
│       └── cli.py
└── parity/
    └── qwen3_6_27b_rtx5090/

src/
└── artifact/
    ├── reader.h
    ├── reader.cpp
    └── storage_layouts.cpp
```

The reference implementation depends on `tools.artifact`, never on converter implementation. The
converter recipe and reference binder independently implement their side of the model-specific
contract. A future checkpoint receives a sibling implementation rather than entering a generic
Python `forward()` framework.

## Phase 0: narrow the specifications and freeze the contract

Before implementation:

1. Add `docs/ninfer-storage-layouts.md` with exact byte and size semantics for the two layouts and
   `raw-bytes-v1`.
2. Add `docs/qwen3.6-27b-ninfer-artifact.md` with separate artifact-contract and conversion-recipe
   sections.
3. Update the registry ledger in `docs/ninfer-container-format.md`.
4. Freeze the canonical object namespace, complete inventory, logical views, frontend resources,
   and full-only route.
5. Record the current q5090 reference Text/Vision performance and memory baseline before moving
   the implementation.

The accepted container shape remains prefix + JSON + payload. Narrow the current container text so
implementation follows the trusted-project engineering principles:

Keep checks that prevent loading the wrong bytes:

- required fields and JSON types;
- closed member sets for the JSON root and each object kind, so model recipes and execution facts
  cannot drift back into the common artifact directory;
- unique object names;
- registered format/layout/encoding identities;
- valid, non-overlapping in-file spans;
- layout-derived encoded size equal to `bytes`;
- complete target inventory and views in the target binder.

Remove requirements that solve no in-scope problem:

- a custom JSON token parser or duplicate-key attack model;
- canonical JSON as an artifact-validity or reproducibility condition;
- immutable-input transaction and atomic-publication protocols;
- failure-residue and interrupted-write behavior;
- exhaustive malformed-input, fuzz, or resource-exhaustion matrices;
- fixed hashes, clean worktrees, or cross-machine byte equality.

The writer may emit stable compact JSON as a simple implementation choice. That spelling is not an
artifact identity or runtime validity requirement.

Phase 0 is complete when the documentation alone determines all 1172 objects and both layout size
functions without introducing source mapping or model execution into the container.

## Phase 1: implement the minimal Python artifact layer

The generic reader surface is deliberately small:

```python
Artifact.open(path)
artifact.model_id
artifact.objects
artifact.find(name)
artifact.payload(object) -> memoryview
artifact.close()
```

Generic descriptors contain only container fields:

```python
TensorObject(name, shape, format, layout, offset, bytes)
ResourceObject(name, encoding, offset, bytes)
```

Implement:

- prefix and JSON encode/decode with the Python standard library;
- payload-relative object placement;
- mmap reading and a name index;
- both layout size functions, encode/decode, and row slicing;
- generic inspection summaries;
- a streaming writer that computes every offset and byte count before encoding payloads.

The writer writes the planned prefix/JSON/padding and then writes each payload directly in order. It
does not implement an atomic publication or interrupted-output protocol; rerunning the converter
overwrites an incomplete output.

Necessary permanent checks are:

- one small artifact containing the three direct and four quantized formats plus one resource;
- exact direct-word preservation;
- Q4/Q5/Q6/W8 code-and-scale agreement with the existing independent packing oracle;
- writer/reader/inspect round trip;
- no q5090 module, source-kind, fusion, or checkpoint concept in the common package.

The last boundary is enforced by package dependencies and API review, not by source-string scanning
tests.

## Phase 2: implement the 27B inventory, recipe, and converter

Migrate, without changing their model meaning:

- the 64-layer Text plan;
- the one-layer MTP plan;
- the 27-layer Vision plan;
- interleaved attention query/gate extraction;
- GDN Q/K/V decomposition and runtime-native convolution transform;
- fused gate/up and shared-input projections;
- Vision patch reshaping;
- current Q4/Q5/Q6/W8 assignments;
- the Q4 draft shortlist and I32 ID map.

Keep `inventory.py` free of source-checkpoint names. Keep source tensor keys, slicing, transforms,
concatenation, and derived values in `recipe.py`. Converter preflight proves that every inventory
object has exactly one recipe and that every required source tensor exists.

The converter performs:

1. exact source-config checks;
2. complete inventory and recipe construction;
3. frontend-resource reads;
4. exact encoded-size and offset planning;
5. prefix and JSON emission;
6. one-object-at-a-time source loading, transform, quantization/encoding, write, and release;
7. an external descriptive conversion report.

Canonical command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_27b_rtx5090.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b_rtx5090.ninfer
```

Use physical order `frontend resources -> Text -> draft head -> MTP -> Vision`.

Write a sidecar at:

```text
out/qwen3_6_27b_rtx5090.ninfer.conversion.json
```

It records descriptive provenance only: source path, actual arguments, model/target identities,
recipe identity, source config summary, converter revision/environment, object and byte summaries,
elapsed time, and final size. It does not participate in artifact loading and requires no fixed
hash, clean worktree, or byte-identical rerun.

The converter emits no `.qus`, accepts no `.qus`, and stores no source tensor, recipe, module kind,
source kind, fusion record, kernel choice, GPU profile, or runtime repack instruction in JSON.

## Phase 3: generate and verify one real artifact immediately

After the converter is functional, generate the complete approximately 17 GiB artifact instead of
continuing development only against synthetic fixtures.

The target verifier checks actual wrong-weight risks:

- complete inventory, names, shapes, formats, layouts, offsets, and sizes;
- all logical row views;
- direct tensors against the transformed source words;
- representative first/middle/last rows and groups for every quantized object;
- every split, concat, reshape, transpose, and derived-value category;
- draft-head ID map and selected source rows;
- all frontend resources;
- isolated `AutoProcessor` and `GenerationConfig` construction from extracted resources.

Do not compare `.qus` and `.ninfer` files, hash the complete artifact, requantize every full matrix a
second time, or compare probabilistic outputs exactly.

The local uncommitted deliverables are:

```text
out/qwen3_6_27b_rtx5090.ninfer
out/qwen3_6_27b_rtx5090.ninfer.conversion.json
```

## Phase 4: implement a narrow C++ reader

Only after the Python bytes and a real artifact exist, add an internal `ninfer_artifact` component.
It uses the existing `nlohmann::json` dependency and implements:

- RAII file ownership;
- prefix and JSON parsing;
- tensor/resource descriptors and a name index;
- absolute payload spans;
- registered-layout alignment and encoded-size calculations.

It does not implement a C++ writer, device allocation, H2D, a materializer, a 27B binder,
WeightStore, target registration, a Program, or q5090 fallback.

Acceptance:

- C++ reads the small Python-written fixture;
- Python and C++ report the same model ID and 1172 real-artifact descriptors;
- the current `ninfer_core`, Engine, and `.qus` path remain unchanged.

## Phase 5: move the existing reference weight boundary and run Text

Preserve the current reference implementation's computation and performance mechanisms:

- the complete 64-layer Text schedule;
- GQA, GDN, MLP, KV, and recurrent-state mathematics;
- chunked prefill and current sampling;
- decoded/packed/streamed residency planning;
- GPU-resident packed payloads and compiled codec;
- decoded control cache;
- embedding row gather;
- chunked full lm-head projection.

The target-private binder consumes the generic directory once and constructs bound physical blocks
and logical row views for Text, MTP, draft head, Vision, and frontend resources. It validates the
exact model ID, complete inventory, shape/format/layout assignments, and row/chunk addressability.
Artifact strings and JSON are resolved only at binding time; the inference hot path uses bound
objects rather than artifact-directory lookups.

WeightStore keeps physical blocks as residency/cache units. Only its q5090 `Block/View` input seam is
replaced. It must not introduce a long-lived dense copy, per-linear CPU payload read, per-token JSON
parse, or repeated payload copy.

First acceptance uses `--ids` so tokenizer/frontend changes do not mask the storage seam:

- Text prefill completes;
- multi-token ordinary generation completes;
- no q5090 header, flag, module, segment, fusion, or source-kind dependency remains;
- the prior decoded/packed/streamed strategy and steady-state decode behavior remain intact.

## Phase 6: use library frontends and restore complete Vision

At cold load, materialize the six artifact resources into a temporary local directory and use:

- `AutoProcessor.from_pretrained(..., local_files_only=True)`;
- the processor tokenizer;
- `apply_chat_template()`;
- `GenerationConfig`.

Delete the new path's hand-written single-turn chat renderer, custom tokenizer semantics, and
external `--processor /path/to/checkpoint` requirement.

Keep checkpoint-private multimodal semantics that Transformers does not replace for the reference
schedule: `mm_token_type_ids`, three-axis MRoPE, `rope_delta`, placeholder/embedding alignment, and
visual-embedding scatter.

VisionEncoder changes only its weight source. Preserve the complete 27-layer tower, independent
media attention segments, position interpolation, Vision RoPE, merger, multiple image/video items,
and the memory lifecycle that executes Vision first, releases its transient weights/workspace, and
then prepares Text weights.

Acceptance covers:

- plain text chat;
- one image;
- one video;
- multiple or mixed image/video input;
- processor -> Vision -> merger -> Text scatter -> MRoPE -> decode;
- no access to the original BF16 checkpoint directory during reference inference.

## Phase 7: connect complete MTP generation

This is the only deliberate model-level completion beyond replacing the artifact boundary. The
current Python reference implements MTP forward and sequential target verification, but its
`generate()` and CLI still use ordinary token-by-token decode.

Use the existing Python MTP mathematics and the current C++ round semantics to implement:

1. shifted MTP prefill after the first target token;
2. persisted MTP hidden state and `k=1..5` greedy proposals;
3. sequential target verification of `current, d1, d2, ...`;
4. longest accepted draft prefix;
5. rejection correction or all-accepted bonus token;
6. committed Text/MTP KV and recurrent-state progression;
7. next-round proposal-state reconstruction;
8. near-capacity ordinary-decode fallback;
9. stop/output-limit handling inside a returned round;
10. acceptance and fallback statistics.

The Python reference remains a clear sequential correctness oracle. Do not copy the C++ fused
small-T kernels, CUDA Graphs, GDN slot framework, or large recurrent snapshots. Execute only the
target prefix that is actually committed and rewind/overwrite the target-private MTP KV cursor as
needed.

Sampling follows the current product semantics:

- draft proposals are greedy one-hot;
- target distributions use temperature, top-k, top-p, presence, and frequency policy;
- the distribution for verification column `i` applies penalties from committed history plus the
  earlier draft tokens provisionally accepted in columns `0..i-1` of the same round;
- accept a draft using its target probability;
- on rejection, sample from the target residual with the draft removed;
- when all drafts accept, sample the final-column bonus;
- provisional same-round tokens affect only that column's temporary distribution; update permanent
  occurrence state only for committed output tokens, never unverified proposals.

Complete multimodal MTP must also repair the known shifted-input error. If position `p+1` is an
image/video placeholder replaced by a Vision merger column, the MTP input paired with target hidden
state `p` uses that composed embedding, not a fresh placeholder-token table lookup. Construct this
per prefill chunk with one-column lookahead instead of retaining an entire long-prompt `[T,5120]`
buffer.

Acceptance covers `k=1` and the primary `k`, accepted/rejected/all-accepted rounds, full and draft
proposal heads, greedy and sampling modes, capacity fallback, stop in the middle of a round, and
Text/Vision combined with MTP. No unverified proposal may enter output.

## Phase 8: final verification, performance, cleanup, and documentation

Keep only permanent tests that protect real contracts:

- one small `.ninfer` artifact containing all seven numeric formats and a resource;
- row-split pack/decode semantics;
- one Python-written/C++-read fixture;
- the 27B inventory and representative fused views;
- query/gate splitting, GDN convolution transform, and draft ID-map mapping;
- MTP acceptance/residual-sampling mathematics;
- multimodal composed-embedding alignment.

Real acceptance runs cover Text, image, video, mixed media, MTP, multimodal+MTP, and real-artifact
inspection. They are not expanded into a large committed fixture matrix.

Run a same-machine before/after comparison on RTX 5090 with the same prompt, context, memory,
prefill-chunk, KV, and codec settings. Use three-run medians for Text prefill, ordinary decode,
Vision latency, and peak GPU memory. A stable regression beyond approximately 10 percent is an
investigation trigger for hot-path name lookup, CPU copying, changed residency, or changed codec
selection; it is not a fragile CI threshold. The newly connected sequential Python MTP path reports
its performance but is not required to match the C++ fused-round speedup.

At the end of this phase:

- the new converter/reference path has no `tools.q5090.*` dependency;
- the useful existing reference implementation lives only in the new target-private directory;
- the old q5090 Python reference namespace and compatibility shims are deleted;
- the hand-written reference chat/tokenizer path is deleted;
- relevant numerical and diagnostic tools move to the new target path.

The current `.qus` converter/reader, C++ q5090 parser/WeightStore, current `.qus` artifact, C++ q5090
tests, and `docs/q5090_packed_file_format_v4.md` remain until the C++ Engine migration. They are not a
new-format compatibility path; they remain the only input route of the still-unmigrated current
Engine. The future Engine cutover deletes that complete legacy route in one step.

README and active documentation must state exactly:

- `.ninfer` conversion, generic reading, and complete Python reference inference are implemented;
- the current C++ Engine still temporarily consumes `.qus`;
- C++ `.ninfer` execution and the multi-target Engine remain pending.

## Completion criteria

This phase is complete only when all of the following hold:

- the selected BF16 checkpoint directly produces one complete `.ninfer` artifact;
- artifact JSON contains only generic directory information and real persistent objects;
- Python and C++ generic readers both consume the artifact;
- the 27B binder restores every physical block and logical view;
- reference inference needs no original checkpoint directory;
- Text generation works end to end;
- image and video Vision generation work end to end;
- MTP proposal, verify, acceptance, correction, bonus, and fallback work end to end;
- multimodal MTP consumes composed Vision embeddings and works end to end;
- the existing reference residency and codec performance mechanisms remain;
- no full-model dense copy is introduced;
- no `.qus` compatibility loader, generic Python model framework, C++ Engine refactor, target
  Program, or 35B implementation enters this phase.
