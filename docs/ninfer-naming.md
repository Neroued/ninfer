# NInfer Project And Artifact-Extension Naming Decision

> Status: accepted on 2026-07-13; project-identity migration and the first native `.ninfer`
> converter/reference route were implemented on 2026-07-14. C++ Engine cutover remains pending.
>
> Authority: this document defines only the project name and the filename extension for native
> model artifacts. It is not an artifact-format specification, container design,
> ABI contract, or migration plan.

## Decision

The canonical name of the elevated project is **NInfer**.

The name is read as **N + Infer**. `N` connects the project to Neroued, while `Infer` states its
inference focus directly. This origin does not establish a separate formal long form.

The canonical filename extension for NInfer native model artifacts is **`.ninfer`**, in lowercase
and including the leading dot.

These two spellings are fixed inputs to the accepted container design and project identity.
Changing either one requires an explicit revision of this decision rather than an incidental rename
in implementation work.

## Canonical usage

| Context | Canonical form | Rule |
|---|---|---|
| Project display name | `NInfer` | Preserve this capitalization in titles, prose, and public identity. |
| Native artifact extension | `.ninfer` | Use the exact lowercase extension, including the leading dot. |

This document does not derive code identifiers from the display name or extension. The current
source tree uses `include/ninfer/`, the `ninfer` C++ root namespace, `ninfer`/`ninfer-*` executables,
`ninfer_*` internal targets, and the `NINFER_*` environment-variable prefix. Those are implemented
source/build identities, not meanings carried by `.ninfer`; the accepted
[`ninfer-engine-architecture.md`](ninfer-engine-architecture.md) independently governs the future
engine source boundary.

The `.ninfer` extension is a naming decision, not a description of bytes. By itself it does not
encode or guarantee:

- a format or ABI version;
- a model architecture, checkpoint, or serving identity;
- a tensor inventory, quantization policy, physical layout, or packing scheme;
- a hardware target or runtime implementation;
- compatibility with a particular NInfer binary;
- integrity, authenticity, or successful validation;
- any particular container structure.

The accepted [`ninfer-container-format.md`](ninfer-container-format.md) defines the relevant
semantics and validation rules without inferring them from the extension alone.

## Relationship to the current implementation

The project identity migration is complete:

- the repository and product are named NInfer;
- current binaries, C++ APIs, namespaces, tools, reports, and command examples use the implemented
  NInfer identities;
- current q5090 v4.2 C++ Engine artifacts retain the `.qus` suffix;
- native `.ninfer` conversion, generic Python reading/inspection, a narrow C++ reader, the 27B
  binder/verifier, and complete Python Text/Vision/MTP reference inference are implemented;
- `.ninfer` is not yet a current C++ Engine input;
- [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) remains authoritative for the
  current C++ `.qus` route, while the native format documents govern `.ninfer`;
- the accepted multi-target engine architecture remains pending and does not turn documented
  checkpoints into runtime support.

Renaming an existing `.qus` file to `.ninfer` is not conversion and does not make its bytes a valid
native NInfer artifact. A loader that happens not to inspect filename extensions still interprets
those bytes under the current q5090 v4.2 contract; that behavior does not establish a `.ninfer`
compatibility alias.

## Implementation status and deferred decisions

Container-format questions formerly deferred here are now governed by
[`ninfer-container-format.md`](ninfer-container-format.md). The following naming and migration
items are either fixed by the artifact-toolchain plan or remain explicitly deferred:

| Subject | Required follow-up |
|---|---|
| Whether `NInfer` needs a formal long form or expansion | Decide separately if the public identity requires one. |
| Remaining `q5090` implementation names and code | New `.ninfer` converter/reference code uses the exact-target key `qwen3_6_27b_rtx5090`; current q5090 C++ names remain only until the separate Engine cutover. |
| `.qus` conversion and cutover | The new converter reads the BF16 checkpoint directly and does not convert `.qus`; current C++ `.qus` loading remains until the separate Engine migration removes it. |
| Uppercase or alternate suffix handling | The native reference CLI requires `.ninfer`; the v1 byte reader does not infer framing from a suffix. |
| Artifact basename, sidecars, manifests, and MIME type | The converter writes a descriptive `.conversion.json` sidecar, but no basename, MIME, or sidecar is part of the artifact contract. |
| Distribution package, registry, MIME, and service identifiers not yet exposed by the project | Define only when such a public contract is introduced. |

This naming document itself does not approve a container ABI, version, magic string, model identity,
or loader compatibility policy. The container specification owns the implemented v1 bytes and
excludes a `.qus` reader; the engine architecture owns the pending C++ source/build cutover. The
existing converter-generated `.ninfer` artifact does not need redesign or regeneration merely
because that later Engine migration begins.

## Rationale

- `NInfer` makes inference visible in the name instead of tying the project identity to one model.
- The initial `N` preserves a concise connection to Neroued without placing the full ID in every
  identifier.
- The name is short, readable, and suitable for a project that can qualify more than one explicitly
  supported architecture.
- `.ninfer` aligns the external file extension with the project identity without carrying the
  former project abbreviation into the native format by default.

These are naming goals, not promises about supported models, GPUs, container capabilities, or the
scope of the eventual runtime. This repository decision also does not constitute trademark,
domain-name, or package-registry registration; any public-release clearance is a separate gate.

## Constraint on follow-up work

The container and engine designs cite this document and treat `NInfer` and `.ninfer` as accepted
inputs. Follow-up implementation must preserve their authority boundaries, resolve only the
remaining subjects above, and must not reintroduce `.qus` loader compatibility or rename the
engine's fixed source identities incidentally.
