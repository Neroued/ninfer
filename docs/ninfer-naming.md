# NInfer Project and Artifact Naming Decision

> Status: accepted and implemented.
>
> Authority: this document defines only the project name and native artifact filename extension. It
> does not define artifact bytes, model inventory, runtime support, or source architecture.

## Decision

The canonical project name is **NInfer**.

The name is read as **N + Infer**. `N` connects the project to Neroued, while `Infer` states its
inference focus directly. It has no separate formal long form.

The canonical filename extension for NInfer model artifacts is **`.ninfer`**, lowercase and
including the leading dot.

| Context | Canonical form | Rule |
|---|---|---|
| Project display name | `NInfer` | Preserve this capitalization in titles and prose. |
| Native artifact extension | `.ninfer` | Use the exact lowercase extension. |

Changing either spelling requires an explicit revision of this decision.

## Implemented identities

The current repository consistently uses:

- `NInfer` as the project display name;
- `.ninfer` for runtime model artifacts;
- `ninfer` as the C++ root namespace and primary executable name;
- `include/ninfer/` for the installed product API;
- `ninfer-*` and `ninfer_*` for executable and internal CMake target names;
- `NINFER_*` for product environment variables;
- exact target keys such as `qwen3_6_27b_rtx5090` for compiled checkpoint/GPU packages and
  target-keyed offline tools.

The currently registered runtime consumes `qwen3_6_27b_rtx5090.ninfer`. Converter, reference,
parity, C++ Engine, CLI, server, benchmark, and diagnostic paths all use the NInfer identity.

## Extension boundary

The `.ninfer` suffix names the project artifact class; it does not itself encode:

- a container version or byte layout;
- a checkpoint architecture or tensor inventory;
- a quantization recipe or storage layout;
- a hardware target or runtime compatibility decision;
- a serving model ID.

Those meanings are defined by the container, numeric-format, storage-layout, exact-target artifact,
and compiled registry contracts. In particular, runtime support follows the complete artifact
metadata/object signature plus the actual GPU and registered target package, not the filename stem.

## Related authority

- [`ninfer-container-format.md`](ninfer-container-format.md) defines `.ninfer` v1 bytes and object
  directory semantics.
- [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md) and
  [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md) define registered persistent formats and
  layouts.
- [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md) defines the registered 27B
  inventory and conversion/binding obligations.
- [`ninfer-engine-architecture.md`](ninfer-engine-architecture.md) defines target selection and the
  source/runtime ownership boundary.

## Rationale

- `NInfer` makes inference visible instead of tying the project identity to one model.
- The initial `N` preserves a concise connection to Neroued.
- The name remains short and readable as more exact targets are separately qualified.
- `.ninfer` aligns the external artifact name with the project identity without encoding a specific
  checkpoint or GPU in the extension itself.

Future package, registry, MIME, or service identifiers should be named only when the project exposes
such a contract. They are not implied by this decision.
