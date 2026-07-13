# NInfer Project And Artifact-Extension Naming Decision

> Status: accepted on 2026-07-13; naming migration has not started.
>
> Authority: this document defines only the future project name and the filename extension reserved
> for future native model artifacts. It is not an artifact-format specification, container design,
> ABI contract, or migration plan.

## Decision

The canonical name of the elevated project is **NInfer**.

The name is read as **N + Infer**. `N` connects the project to Neroued, while `Infer` states its
inference focus directly. This origin does not establish a separate formal long form.

The canonical filename extension reserved for future NInfer native model artifacts is
**`.ninfer`**, in lowercase and including the leading dot.

These two spellings are fixed inputs to the accepted container design and the future project
migration. Changing either one requires an explicit revision of this decision rather than an
incidental rename in an implementation plan.

## Canonical usage

| Context | Canonical form | Rule |
|---|---|---|
| Project display name | `NInfer` | Preserve this capitalization in titles, prose, and public identity. |
| Future artifact extension | `.ninfer` | Use the exact lowercase extension, including the leading dot. |

No binary name, C++ namespace, include path, package name, environment-variable prefix, service
identifier, repository directory, or executable target is assigned by this document. Their future
spellings, including any bare lowercase project identifier, must be chosen as part of the migration
design; they are not implied by the `.ninfer` extension.

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

This decision does not rename or modify the implemented system. Until a separate migration lands:

- the repository and current product remain `qwen3.6-ultraspeed` and QUS where the source tree says
  so;
- current binaries, C++ APIs, namespaces, tools, reports, and command examples retain their existing
  names;
- current q5090 v4.2 artifacts retain the `.qus` suffix;
- `.ninfer` is not a current implemented runtime input; its accepted v1 contract remains pending
  implementation;
- [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) remains the only normative
  contract for the currently implemented artifact format.

Renaming an existing `.qus` file to `.ninfer` is not conversion and does not make its bytes a future
NInfer format. A loader that happens not to inspect filename extensions still interprets those bytes
under the current q5090 v4.2 contract; that behavior does not establish a `.ninfer` compatibility
alias.

Existing future-facing plans may still contain candidate names such as `QUS packed-container v5`,
future `.qus` filenames, or QUS-derived wire strings. Those passages predate this decision and do
not override `NInfer` or the `.ninfer` extension. They are deliberately not replaced mechanically:
the accepted [`ninfer-container-format.md`](ninfer-container-format.md) owns the future container
name, JSON identifier namespaces, and model/container boundary while treating this document as its
naming input.

## Deferred decisions

Container-format questions formerly deferred here are now governed by
[`ninfer-container-format.md`](ninfer-container-format.md). The following naming and migration
questions remain explicitly undecided:

| Subject | Required follow-up |
|---|---|
| Whether `NInfer` needs a formal long form or expansion | Decide separately if the public identity requires one. |
| Remaining `q5090` implementation names and code | Decide during the implementation migration; `.ninfer` v1 has no q5090-style target-profile field. |
| `.qus` compatibility and cutover | Define in a separate migration and compatibility decision. |
| Uppercase or alternate suffix handling | Define CLI/path policy during implementation; filename spelling is not a v1 byte-parsing input. |
| Artifact basename, sidecars, manifests, and MIME type | Define only if those contracts are needed. |
| Repository, binary, namespace, include, CLI, and service renaming | Define and inventory in the migration plan. |

This naming document itself does not approve a container ABI, version, magic string, model identity,
or compatibility policy. The separate container specification now owns the first four subjects;
migration compatibility remains undecided.

## Rationale

- `NInfer` makes inference visible in the name instead of tying the project identity to one model.
- The initial `N` preserves a concise connection to Neroued without placing the full ID in every
  identifier.
- The name is short, readable, and suitable for a project that can qualify more than one explicitly
  supported architecture.
- `.ninfer` aligns the external file extension with the project identity without carrying the old
  QUS name into the future format by default.

These are naming goals, not promises about supported models, GPUs, container capabilities, or the
scope of the eventual runtime. This repository decision also does not constitute trademark,
domain-name, or package-registry registration; any public-release clearance is a separate gate.

## Constraint on follow-up work

The container design cites this document and treats `NInfer` and `.ninfer` as accepted inputs. The
remaining project-migration plan must do the same, state its own authority, and resolve the remaining
subjects above without silently expanding the scope of this naming decision.
