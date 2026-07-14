# NInfer Project And Artifact-Extension Naming Decision

> Status: accepted on 2026-07-13; project-identity migration implemented on 2026-07-14. The
> `.ninfer` artifact route is under implementation.
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
| Future artifact extension | `.ninfer` | Use the exact lowercase extension, including the leading dot. |

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
- current q5090 v4.2 artifacts retain the `.qus` suffix;
- `.ninfer` is not a current C++ runtime input; its accepted v1 toolchain is under implementation;
- [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) remains the only normative
  contract for the currently implemented artifact format;
- the accepted multi-target engine architecture remains pending and does not turn documented
  checkpoints into runtime support.

Renaming an existing `.qus` file to `.ninfer` is not conversion and does not make its bytes a future
NInfer format. A loader that happens not to inspect filename extensions still interprets those bytes
under the current q5090 v4.2 contract; that behavior does not establish a `.ninfer` compatibility
alias.

## Implementation status and deferred decisions

Container-format questions formerly deferred here are now governed by
[`ninfer-container-format.md`](ninfer-container-format.md). The following naming and migration
items are either fixed by the artifact-toolchain plan or remain explicitly deferred:

| Subject | Required follow-up |
|---|---|
| Whether `NInfer` needs a formal long form or expansion | Decide separately if the public identity requires one. |
| Remaining `q5090` implementation names and code | New `.ninfer` converter/reference code uses the exact-target key `qwen3_6_27b_rtx5090`; current q5090 C++ names remain only until the separate Engine cutover. |
| `.qus` conversion and cutover | The new converter reads the BF16 checkpoint directly and does not convert `.qus`; current C++ `.qus` loading remains until the separate Engine migration removes it. |
| Uppercase or alternate suffix handling | Define CLI/path policy during implementation; filename spelling is not a v1 byte-parsing input. |
| Artifact basename, sidecars, manifests, and MIME type | Define only if those contracts are needed. |
| Distribution package, registry, MIME, and service identifiers not yet exposed by the project | Define only when such a public contract is introduced. |

This naming document itself does not approve a container ABI, version, magic string, model identity,
or loader compatibility policy. The container specification owns the v1 bytes and excludes a
`.qus` reader; the engine architecture owns its future source/build identities. Implementing those
designs requires a separate scoped migration and regenerated artifacts.

## Rationale

- `NInfer` makes inference visible in the name instead of tying the project identity to one model.
- The initial `N` preserves a concise connection to Neroued without placing the full ID in every
  identifier.
- The name is short, readable, and suitable for a project that can qualify more than one explicitly
  supported architecture.
- `.ninfer` aligns the external file extension with the project identity without carrying the
  former project abbreviation into the future format by default.

These are naming goals, not promises about supported models, GPUs, container capabilities, or the
scope of the eventual runtime. This repository decision also does not constitute trademark,
domain-name, or package-registry registration; any public-release clearance is a separate gate.

## Constraint on follow-up work

The container and engine designs cite this document and treat `NInfer` and `.ninfer` as accepted
inputs. Follow-up implementation must preserve their authority boundaries, resolve only the
remaining subjects above, and must not reintroduce `.qus` loader compatibility or rename the
engine's fixed source identities incidentally.
