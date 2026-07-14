# NInfer Naming Cutover Implementation Plan

Status: implementation in progress; Phases 0-3 complete

## 1. Goal

Replace the current QUS and `qwen3.6-ultraspeed` project identity with NInfer without changing
inference behavior. The completed cutover must leave one coherent identity across the repository,
source API, build products, project-owned tooling, active documentation, GitHub repository, and
canonical local checkout:

```text
display name:       NInfer
GitHub repository:  Neroued/ninfer
local checkout:     /home/neroued/ninfer
source stem:        ninfer
```

This is a breaking rename of project-owned names, not a compatibility migration. The implementation
must replace the old names directly and must not add forwarding headers, namespace aliases, old
binary symlinks, duplicate CMake targets, Python import shims, environment-variable fallbacks, or
report-schema aliases.

The work is delivered through three separately observable cutover boundaries:

1. prepare and verify all tracked repository changes while the checkout and GitHub repository still
   have their old names;
2. rename the existing GitHub repository in place and update every maintained remote reference;
3. move the existing local checkout in place, rebuild path-sensitive local state, and verify from
   the new canonical path.

All three boundaries belong to one naming milestone, but they must not be collapsed into one
unobservable operation.

## 2. Meaning of no functional change

The following behavior must remain unchanged:

- model mathematics, tensor values, rounding, and numerical tolerances;
- CUDA kernels, dispatch policy, schedules, CUDA Graph behavior, and memory policy;
- q5090 v4.2 bytes, layouts, manifests, parser behavior, and loading requirements;
- Text, MTP, Vision, sampling, prefix reuse, and generation semantics;
- CLI option meanings and default values;
- OpenAI and Anthropic request, response, streaming, finish, and tool-call semantics, except for a
  project-brand value explicitly listed below;
- benchmark measurement logic and performance methodology;
- the supported checkpoint, GPU, workload, and current runtime capabilities.

The following are intentional name-contract changes within this plan:

- public C++ include paths and namespaces;
- CMake project, target, test, and compile-definition names;
- executable and repository-local Python package names;
- project-owned environment-variable names;
- log, error, profiler-range, and help-text prefixes;
- project-owned machine-readable tool, report, corpus, and activation-dump discriminators;
- project-branded CSV field names;
- the OpenAI model-listing `owned_by` value.

These are observable breaking changes, but they carry no old-name compatibility. Existing schema,
CLI, and report tests must be updated to protect the new names and unchanged surrounding behavior.

## 3. Non-goals

- Do not implement the accepted NInfer engine architecture, source tree, target packages, or
  container loader in this change.
- Do not add Qwen3.6-35B-A3B runtime support.
- Do not redesign current `core`, `kernels`, `model`, `runtime`, `text`, `media`, `serve`, `tools`,
  `bench`, or `eval` ownership.
- Do not change model, kernel, serving, evaluation, or benchmark behavior while touching names.
- Do not rename `.qus` artifacts to `.ninfer` or teach either loader to accept the other format.
- Do not regenerate, cut over, or delete current `.qus` artifacts. Those actions belong to the
  later container implementation migration, not to completion of project naming.
- Do not modify q5090 magic, layout names, type names, object names, or conversion behavior.
- Do not rewrite archived QUS-era documents as though they were written for NInfer.
- Do not rename the GitHub owner `Neroued`, the default branch `master`, Qwen checkpoint identities,
  or unrelated external dependencies.
- Do not create a second GitHub repository and copy history into it.
- Do not retain a permanent `/home/neroued/qwen3.6-ultraspeed` symlink after the local move.
- Do not combine this work with cleanup, formatting, architecture refactoring, or test expansion.

## 4. Canonical rename matrix

| Surface | Current | Final |
|---|---|---|
| Display name | QUS / `qwen3.6-ultraspeed` | `NInfer` |
| GitHub repository | `Neroued/qwen3.6-ultraspeed` | `Neroued/ninfer` |
| Local checkout | `/home/neroued/qwen3.6-ultraspeed` | `/home/neroued/ninfer` |
| CMake project | `qus` | `ninfer` |
| Public include root | `include/qus/` | `include/ninfer/` |
| C++ root namespace | `qus` | `ninfer` |
| Core library | `qus_core` | `ninfer_core` |
| Serving library | `qus_serve` | `ninfer_serve` |
| Main executable | `qus` | `ninfer` |
| Serving executable | `qus-serve` | `ninfer-serve` |
| Preprocess diagnostic | `qus-preprocess` | `ninfer-preprocess` |
| Vision diagnostic | `qus-vision-dump` | `ninfer-vision-dump` |
| Benchmark stem | `qus_bench`, `qus_*_bench` | `ninfer_bench`, `ninfer_*_bench` |
| CTest stem | `qus_*_test` | `ninfer_*_test` |
| Evaluation package | `qus_eval` | `ninfer_eval` |
| Evaluation CLI display | `qus-eval` | `ninfer-eval` |
| Macro/environment stem | `QUS_*` | `NINFER_*` |
| Log/NVTX/report stem | `qus` / `qus_*` | `ninfer` / `ninfer_*` |
| OpenAI model owner | `"owned_by": "qus"` | `"owned_by": "ninfer"` |

The executable CMake target is named `ninfer` directly. This plan does not reserve an unused
umbrella library name for a hypothetical future build layout.

## 5. Preserved identities

The following strings are not project-brand leftovers and must remain where their current contracts
require them:

- `.qus` for current q5090 v4.2 artifacts;
- q5090 v4.2 file magic, layout, manifest, tensor, parser, verifier, and converter identities;
- `q5090_w4g64_mixed_v4_2` and the canonical current artifact basename;
- `tools/q5090/`, `tools/q5090_convert/`, and `Q5090*` source types;
- Qwen checkpoint names, model IDs, source filenames, and architecture identifiers;
- historical QUS text, commands, paths, and report names under `docs/archive/`;
- existing ignored profiler reports, evaluation runs, benchmark evidence, and generated artifacts.

The intended intermediate product state after this plan is:

```text
NInfer executable + current q5090 v4.2 .qus artifact
```

The accepted `.ninfer` extension remains reserved for the new container bytes. A `.qus` file is not
converted by changing its suffix, and the current loader must not advertise `.ninfer` support.

## 6. Current facts

Facts recorded during planning on 2026-07-14:

- repository: `/home/neroued/qwen3.6-ultraspeed`;
- branch: `master`;
- planning baseline: `d42b15f60a90c671ee7abdfd04c458524e7e7e0e`;
- local `master` and `origin/master` are currently equal at that commit;
- origin: `https://github.com/Neroued/qwen3.6-ultraspeed.git`;
- the GitHub repository is private, has one collaborator, and uses `master` as its default branch;
- GitHub repository numeric ID: `1281510950`; node ID: `R_kgDOTGJOJg`;
- `Neroued/ninfer` does not currently exist;
- there are no checked-in GitHub Actions workflows, Pages site, releases, tags, webhooks, deploy
  keys, deployments, environments, Actions secrets, or Actions variables;
- the GitHub UI reports only its dynamic Dependency Graph workflow;
- the current token cannot enumerate GitHub Packages, so the Packages page requires a manual
  execution-day check;
- `/home/neroued/ninfer` does not currently exist;
- the checkout is approximately 25 GiB, including approximately 17 GiB in `out/`, 1.3 GiB in
  `build/`, and 649 MiB in `.git/`;
- `build/`, `compile_commands.json`, and `eval/.venv/` contain old absolute paths;
- a local `feat/lm-head-draft-shortlist` branch has no configured remote tracking branch and must be
  included in the external Git bundle even though it is outside the naming change;
- excluding third-party and archived material, the old project identity appears across hundreds of
  tracked files, including 57 public headers, the C++ namespace, CMake targets, benchmarks, tests,
  tools, and the `eval/qus_eval` package;
- `.qus` appears throughout active format, fixture, command, and artifact contracts and cannot be a
  blind replacement target.

These facts are a planning baseline, not permission to skip execution-day checks.

## 7. Focused reading list

Read these sources before changing the corresponding surface:

- [`../ninfer-naming.md`](../ninfer-naming.md) — accepted `NInfer` and `.ninfer` spelling and the
  boundary between project naming and container bytes;
- [`../ninfer-project-positioning.md`](../ninfer-project-positioning.md) — project purpose and the
  wording that the new README must reflect;
- [`../ninfer-engine-architecture.md`](../ninfer-engine-architecture.md) — accepted future
  `include/ninfer/`, namespace, component stem, and source/build boundary;
- [`../ninfer-container-format.md`](../ninfer-container-format.md) — why `.ninfer` cannot alias the
  current q5090 format;
- [`../q5090_packed_file_format_v4.md`](../q5090_packed_file_format_v4.md) — current `.qus` contract
  that this change must preserve;
- [`../README.md`](../README.md), [`../../README.md`](../../README.md), and
  [`../../AGENTS.md`](../../AGENTS.md) — active navigation, current product claims, local paths, and
  repository rules;
- [`../serving.md`](../serving.md), [`../../bench/README.md`](../../bench/README.md), and
  [`../../eval/README.md`](../../eval/README.md) — user-visible command and schema surfaces;
- [GitHub repository-renaming documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository)
  and [remote URL documentation](https://docs.github.com/en/get-started/git-basics/managing-remote-repositories)
  — external cutover behavior and limitations;
- [GitHub OIDC reference](https://docs.github.com/en/actions/reference/security/oidc) and
  [organization-ruleset documentation](https://docs.github.com/en/organizations/managing-organization-settings/creating-rulesets-for-repositories-in-your-organization)
  — execution-day checks for integrations that can depend on repository identity.

Read archived files only to classify historical references; do not maintain their old commands.

## 8. Ownership and coordination

The rename follows current ownership only for determining which verification protects each surface:

- root and `src/` CMake own project, library, executable, and link identities;
- `include/`, `src/`, `tests/`, and `bench/` jointly own the compiled C++ namespace/include closure;
- `bench/`, `tools/bench/`, and their tests own benchmark executables and report contracts;
- `eval/` owns the Python package, CLI display, configurations, and evaluation documentation;
- `serve/`, serving tests, and `docs/serving.md` jointly own protocol-visible branding;
- `README.md`, `docs/README.md`, `AGENTS.md`, and the accepted NInfer documents own project claims
  and authority boundaries;
- GitHub settings own the repository slug and description;
- the local environment owns the checkout path, derived build tree, virtual environment, ignored
  artifacts, profiler output, and editor state.

No other source-editing work may run concurrently with Phases 1 through 4. All agents, IDE build
tasks, servers, profilers, and shells using the old checkout as their current directory must be
quiescent before the local move.

## 9. Dependencies and sequence

The mandatory order is:

```text
protect baseline
      -> tracked runtime/build rename
      -> tooling and machine-readable rename
      -> active documentation rename
      -> clean-build and behavior-equivalence review
      -> merge and push through old GitHub identity
      -> GitHub repository rename and remote update
      -> local checkout move
      -> rebuild path-sensitive state and final gate
      -> archive plan and evidence
```

Do not rename GitHub before all tracked changes pass. Do not move the checkout before the GitHub
repository and local remote use the final identity. Do not archive this plan until the new-path gate
passes.

## 10. Phase 0 — Protect and record the baseline

### Work

1. Stop concurrent source edits and confirm one clean worktree.
2. Recheck the destination repository and local path are still unused.
3. Fetch the old origin and require local `master` to equal `origin/master`.
4. Record all refs, GitHub repository metadata, relevant settings, and disk state.
5. Create a verified Git bundle outside the checkout containing all refs, including local-only
   branches.
6. Rebuild the baseline from the current HEAD, then record the canonical artifact identity and a
   fixed deterministic CLI result. Do not trust a pre-existing binary without rebuilding it.
7. Create `refactor/ninfer-naming` from the synchronized baseline.

### Commands

```bash
(
set -euo pipefail
cd /home/neroued/qwen3.6-ultraspeed
git status --short --branch
test -z "$(git status --porcelain)"
git fetch --prune origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
test ! -e /home/neroued/ninfer
if gh repo view Neroued/ninfer --json nameWithOwner >/dev/null 2>&1; then
  echo 'Neroued/ninfer already exists' >&2
  exit 1
fi

mkdir -p /home/neroued/backups
git show-ref > /home/neroued/backups/ninfer-pre-cutover-refs.txt
git bundle create /home/neroued/backups/ninfer-pre-cutover.bundle --all
git bundle verify /home/neroued/backups/ninfer-pre-cutover.bundle
eval/.venv/bin/python --version \
  > /home/neroued/backups/ninfer-pre-cutover-eval-python.txt
eval/.venv/bin/python -m pip --version \
  > /home/neroued/backups/ninfer-pre-cutover-eval-pip.txt
eval/.venv/bin/python -m pip freeze --all \
  > /home/neroued/backups/ninfer-pre-cutover-eval-requirements.txt

gh api repos/Neroued/qwen3.6-ultraspeed \
  --jq '{id,node_id,full_name,private,default_branch,has_pages,open_issues_count}' \
  > /home/neroued/backups/ninfer-pre-cutover-github.json

WEIGHTS=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
stat -c '%d:%i:%s' "$WEIGHTS" \
  > /home/neroued/backups/ninfer-pre-cutover-artifact.stat
sha256sum "$WEIGHTS" \
  > /home/neroued/backups/ninfer-pre-cutover-artifact.sha256

test ! -e cmake-build-qus-baseline
cmake -S . -B cmake-build-qus-baseline -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build cmake-build-qus-baseline -j
test "$(ctest --test-dir cmake-build-qus-baseline -N \
  | awk '/Total Tests:/ {print $3}')" = 54
ctest --test-dir cmake-build-qus-baseline --output-on-failure \
  -E '^qus_engine_mtp_e2e_test$'
if ctest --test-dir cmake-build-qus-baseline --output-on-failure \
    -R '^qus_engine_mtp_e2e_test$' 2>&1 \
    | tee /home/neroued/backups/ninfer-pre-cutover-known-mtp-failure.log; then
  echo 'known baseline MTP test unexpectedly passed' >&2
  exit 1
fi
rg -q 'partial reuse turn 2 parity differs \(mtp-off\)' \
  /home/neroued/backups/ninfer-pre-cutover-known-mtp-failure.log
rg -q 'partial reuse turn 2 parity differs \(mtp-on\)' \
  /home/neroued/backups/ninfer-pre-cutover-known-mtp-failure.log

./cmake-build-qus-baseline/src/qus "$WEIGHTS" \
  --prompt '用一句话解释 prefill。' --no-thinking --greedy --max-new 16 \
  > /home/neroued/backups/ninfer-pre-cutover-greedy.txt

git switch -c refactor/ninfer-naming
)
```

`gh repo view Neroued/ninfer` is expected to fail while the destination is unused; success is a stop
condition requiring investigation. Do not choose a different repository name implicitly.

Execution established two pre-existing baseline facts that the remaining gates must preserve rather
than conceal. A clean configure must pass `CMAKE_CUDA_ARCHITECTURES=120a`; the current top-level
CMake ordering otherwise lets CMake cache architecture `75` before the project default is applied.
Also, 53 of 54 CTest entries pass, while `qus_engine_mtp_e2e_test` reproducibly fails only its
partial-reuse turn-2 parity checks for both MTP-off and MTP-on. Fixing either issue is outside this
identity-only change. Every later CTest gate therefore runs all other entries to success, runs the
renamed MTP entry separately, and requires the same two failure markers.

The Phase 0 execution baseline, after this plan itself landed, is
`98e2c459daea408665a2d6882d8d3401367a00d8`. The verified bundle, old `master`, artifact evidence,
and deterministic CLI evidence were captured against that commit.

### Completion gate

- the old checkout remains clean before branch creation;
- all current refs are present in a bundle that passes `git bundle verify`;
- the old origin has the complete baseline;
- the destination names are unused;
- artifact identity and deterministic output evidence exist outside the checkout.

## 11. Phase 1 — Rename the compiled runtime and build identity

### Work

Perform one compiled closure change:

- move `include/qus/` to `include/ninfer/`;
- replace project-owned C++ namespace and qualified-name uses;
- replace public and private include paths;
- rename the CMake project, variables, functions, libraries, executables, tests, benchmarks, compile
  definitions, and target references;
- rename tracked benchmark/test/tool filenames whose basename carries the QUS project identity;
- rename project-branded source comments, diagnostics, log prefixes, help display, and NVTX ranges;
- preserve q5090 names and `.qus` paths;
- make no code-flow, algorithm, layout, allocation, or numerical edits.

The C++ and CMake closure is one delivery boundary. Do not create an intermediate commit where the
include tree, namespace, or build graph is broken.

### Commit

```text
refactor(naming): switch runtime identity to ninfer
```

### Verification

```bash
cmake -S . -B cmake-build-ninfer-rename -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build cmake-build-ninfer-rename -j
test "$(ctest --test-dir cmake-build-ninfer-rename -N \
  | awk '/Total Tests:/ {print $3}')" = 54
ctest --test-dir cmake-build-ninfer-rename --output-on-failure \
  -E '^ninfer_engine_mtp_e2e_test$'
(
  set -euo pipefail
  if ctest --test-dir cmake-build-ninfer-rename --output-on-failure \
      -R '^ninfer_engine_mtp_e2e_test$' 2>&1 \
      | tee /home/neroued/backups/ninfer-phase1-known-mtp-failure.log; then
    echo 'known baseline MTP test unexpectedly passed' >&2
    exit 1
  fi
  rg -q 'partial reuse turn 2 parity differs \(mtp-off\)' \
    /home/neroued/backups/ninfer-phase1-known-mtp-failure.log
  rg -q 'partial reuse turn 2 parity differs \(mtp-on\)' \
    /home/neroued/backups/ninfer-phase1-known-mtp-failure.log
)

test -x cmake-build-ninfer-rename/src/ninfer
test -x cmake-build-ninfer-rename/src/ninfer-serve
test ! -e cmake-build-ninfer-rename/src/qus
test ! -e cmake-build-ninfer-rename/src/qus-serve

cmake-build-ninfer-rename/src/ninfer --help
cmake-build-ninfer-rename/src/ninfer-serve --help
cmake-build-ninfer-rename/bench/ninfer_bench --help
```

The fresh build directory is required so stale QUS outputs from `build/` cannot make the gate pass.

## 12. Phase 2 — Rename tooling and machine-readable project identity

### Work

- move `eval/qus_eval/` to `eval/ninfer_eval/` and update all repository-local imports and module
  invocations;
- rename project-branded benchmark matrix scripts and support filenames;
- rename project-owned environment variables and test-only controls without fallbacks;
- replace benchmark tool, report, corpus, matrix, activation-dump, and related discriminator values;
- replace project-branded CSV fields and producer/consumer references together;
- change the OpenAI model-listing owner from `qus` to `ninfer` without changing the response shape;
- update existing schema, report, Python, and CLI tests for the new names;
- preserve external target labels that describe Qwen or third-party engines rather than this project.

No compatibility test for the removed names may remain.

### Commit

```text
refactor(tooling): switch project tooling to ninfer
```

Any tool required to compile the Phase 1 closure must be included in Phase 1 rather than leaving an
unbuildable boundary. Phase 2 is independently complete when Python and machine-readable consumers
close over the new identity.

### Verification

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
PYTHONPATH=eval "$PYTHON" -m py_compile \
  $(rg --files eval/ninfer_eval -g '*.py')
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'

ctest --test-dir cmake-build-ninfer-rename --output-on-failure \
  -R 'ninfer_(openai_schema|anthropic_schema|bench_support|runtime_file_tap)_test'

PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval --help
```

The main benchmark report continues its existing schema lineage: changing its fixed
`artifact_type`, `tool`, and project-branded field names requires `schema_version` 6 to become 7,
with `bench/README.md`, fixtures, producers, and tests updated together. Other artifacts whose new
discriminator establishes a new NInfer identity retain their existing payload schema version when
the payload fields and meanings are otherwise unchanged; do not bump versions for prose or
executable filenames alone.

## 13. Phase 3 — Establish the active NInfer project identity

### README outcome

The root README must lead with the accepted NInfer position:

> Selected checkpoints. Selected GPUs. Maximum single-GPU inference performance.

It must then distinguish project direction from current implementation facts:

- current delivered target: Qwen3.6-27B on RTX 5090;
- current workload: one active request and one GPU;
- current artifact: q5090 v4.2 `.qus`;
- current commands: the renamed NInfer executables;
- Qwen3.6-35B-A3B architecture documentation is not runtime support;
- the accepted new engine architecture and `.ninfer` container remain pending implementation.

### Documentation work

- update `README.md`, `docs/README.md`, and `AGENTS.md` to the active NInfer project identity;
- set the canonical repository path in `AGENTS.md` to `/home/neroued/ninfer`;
- update `ninfer-naming.md` so its status distinguishes an implemented project identity from the
  still-unimplemented `.ninfer` bytes, loader, and artifact cutover;
- make `ninfer-project-positioning.md` the current project-position authority while preserving its
  explicit prohibition on describing implementation status;
- make `docs/README.md` classify naming and positioning as current project authorities, while
  `ninfer-tensor-formats.md`, `ninfer-container-format.md`, and `ninfer-engine-architecture.md`
  remain accepted decisions pending implementation;
- make README and AGENTS—not the positioning document—state that the current delivered route is
  still Qwen3.6-27B + RTX 5090 + q5090 v4.2 `.qus`;
- update active operational documentation for renamed commands, includes, namespaces, environment
  variables, reports, and tools;
- keep current implementation documents accurate without rewriting their architecture content;
- update or archive an active plan only if its actual lifecycle requires that action;
- leave archived QUS-era technical content unchanged, except for archive navigation if this plan
  later creates a new archive era.

### Commit

```text
docs(project): complete ninfer identity cutover
```

### Documentation verification

```bash
git diff --check

rg -n -i \
  -g '!docs/archive/**' -g '!third_party/**' \
  -g '!build/**' -g '!cmake-build-*/**' -g '!out/**' -g '!profiles/**' \
  '(qwen3\.6-ultraspeed|include/qus|namespace qus|\bQUS\b|\bqus\b)' .

PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
"$PYTHON" - <<'PY'
from pathlib import Path
import re
import subprocess

root = Path.cwd()
missing = []
tracked = subprocess.check_output(
    ["git", "ls-files", "*.md"], text=True
).splitlines()
markdown = [
    root / path for path in tracked
    if not path.startswith("docs/archive/")
    and not path.startswith("third_party/")
]

def prose_without_code(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith("```"):
            fence = "```"
            continue
        if stripped.startswith("~~~"):
            fence = "~~~"
            continue
        lines.append(re.sub(r"`+[^`]*`+", "", line))
    return "\n".join(lines)

for md in markdown:
    text = prose_without_code(md.read_text(encoding="utf-8"))
    for target in re.findall(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)", text):
        target = target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        resolved = (md.parent / target).resolve()
        if not resolved.exists():
            missing.append(f"{md.relative_to(root)} -> {target}")
if missing:
    raise SystemExit("missing local Markdown links:\n" + "\n".join(missing))
PY
```

The `rg` command is a review audit, not a source-scanning unit test. Every result must be classified
as one of:

- a required `.qus`/q5090 current-format identity;
- an intentional historical statement about the former project;
- an archived record excluded by the command;
- an error to fix before the phase completes.

## 14. Phase 4 — Final tracked-tree review and behavior gate

### Independent review

After all three commits exist, perform a separate review from four perspectives:

1. **behavior:** no numerical, scheduling, memory, kernel, sampling, or protocol-semantic change;
2. **contract:** every project-owned name changed once, with producers, consumers, tests, and docs
   agreeing and no compatibility aliases;
3. **format:** q5090 v4.2 and `.qus` bytes, names, verification, and loading remain intact;
4. **documentation:** project direction, current implementation, and pending architecture/container
   work are not conflated.

Review moves and replacements explicitly:

```bash
BASE=$(git merge-base origin/master HEAD)
git diff --find-renames --summary "$BASE"..HEAD
git diff --stat "$BASE"..HEAD
git diff --check "$BASE"..HEAD
git log --oneline "$BASE"..HEAD
```

### Full gate

```bash
cmake --build cmake-build-ninfer-rename -j
test "$(ctest --test-dir cmake-build-ninfer-rename -N \
  | awk '/Total Tests:/ {print $3}')" = 54
ctest --test-dir cmake-build-ninfer-rename --output-on-failure \
  -E '^ninfer_engine_mtp_e2e_test$'
(
  set -euo pipefail
  if ctest --test-dir cmake-build-ninfer-rename --output-on-failure \
      -R '^ninfer_engine_mtp_e2e_test$' 2>&1 \
      | tee /home/neroued/backups/ninfer-final-known-mtp-failure.log; then
    echo 'known baseline MTP test unexpectedly passed' >&2
    exit 1
  fi
  rg -q 'partial reuse turn 2 parity differs \(mtp-off\)' \
    /home/neroued/backups/ninfer-final-known-mtp-failure.log
  rg -q 'partial reuse turn 2 parity differs \(mtp-on\)' \
    /home/neroued/backups/ninfer-final-known-mtp-failure.log
)

PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
PYTHONPATH=eval "$PYTHON" -m py_compile \
  $(rg --files eval/ninfer_eval -g '*.py')
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'

WEIGHTS=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
"$PYTHON" -m tools.q5090_convert.verify "$WEIGHTS" --quick
cmake-build-ninfer-rename/src/ninfer "$WEIGHTS" \
  --prompt '用一句话解释 prefill。' --no-thinking --greedy --max-new 16 \
  > /home/neroued/backups/ninfer-post-rename-greedy.txt
diff -u /home/neroued/backups/ninfer-pre-cutover-greedy.txt \
        /home/neroued/backups/ninfer-post-rename-greedy.txt

git diff --check
git status --short
```

NSYS and NCU are not required because this change does not alter performance implementation. A
deterministic real-artifact smoke, complete execution of the CTest inventory with the one known
entry reproducing the same two recorded failure markers, and diff review are the proportional
behavior-equivalence evidence.

### Merge gate

- every review finding is resolved;
- the three commits are individually coherent and the branch passes as a whole;
- the worktree is clean;
- no GitHub or local-directory cutover has occurred yet.

Fast-forward the complete branch into `master`, push through the old GitHub identity, and require
the pushed SHA to equal local `master` before proceeding:

```bash
git switch master
git merge --ff-only refactor/ninfer-naming
git push origin master
git fetch --prune origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
```

## 15. Phase 5 — Rename the existing GitHub repository

### Preflight

Perform this phase in one short maintenance window:

- confirm `Neroued/ninfer` remains unused;
- confirm the Packages page has no package or image whose naming/access requires a separate change;
- recapture repository ID, HEAD, default branch, Pages, hooks, keys, environments, variables, and
  secrets counts;
- confirm the repository is not published as an Action or reusable workflow;
- check repository and account/organization rulesets, installed GitHub Apps, OIDC configuration and
  external trust policies, and other integrations that select the repository by slug;
- if OIDC is present, make its provider trust both the documented post-rename immutable subject and
  the current subject before renaming, verify the new subject after cutover, and only then remove
  the old trust entry;
- confirm local `master`, old `origin/master`, and the reviewed cutover SHA are equal.

GitHub documents ordinary web and Git redirects after a repository rename, but does not redirect a
GitHub Pages project URL or calls to an Action hosted in the renamed repository. Neither exception
currently applies, but the execution-day check remains mandatory.

### Mutation

Rename the existing repository object; do not create or transfer a second repository:

```bash
(
  set -euo pipefail
  gh api --method PATCH repos/Neroued/qwen3.6-ultraspeed \
    -f name=ninfer \
    -f description='High-performance single-GPU inference for selected model checkpoints and GPUs.'
)
```

The resulting GitHub description is:

```text
High-performance single-GPU inference for selected model checkpoints and GPUs.
```

### Verification and local remote update

```bash
(
  set -euo pipefail
  gh api repos/Neroued/ninfer \
    --jq '{id,node_id,full_name,description,private,default_branch,has_pages,open_issues_count}'
  test "$(gh api repos/Neroued/ninfer --jq '.id')" = 1281510950
  test "$(gh api repos/Neroued/ninfer --jq '.node_id')" = R_kgDOTGJOJg
  test "$(gh api repos/Neroued/ninfer --jq '.description')" = \
    'High-performance single-GPU inference for selected model checkpoints and GPUs.'

  NEW_HEAD=$(git ls-remote https://github.com/Neroued/ninfer.git refs/heads/master | cut -f1)
  test "$NEW_HEAD" = "$(git rev-parse HEAD)"

  git remote set-url origin https://github.com/Neroued/ninfer.git
  git fetch --prune origin
  test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
  git push --dry-run origin master

  OLD_HEAD=$(git ls-remote https://github.com/Neroued/qwen3.6-ultraspeed.git \
    refs/heads/master | cut -f1)
  test "$OLD_HEAD" = "$NEW_HEAD"
  git remote -v
)
```

Require repository numeric ID `1281510950`, node ID `R_kgDOTGJOJg`, and `master` HEAD to remain
unchanged. Recheck the GitHub settings snapshot after the rename.

The old repository slug must never be reused under `Neroued`; GitHub warns that doing so removes the
redirect. The old redirect is recovery support, not a maintained endpoint.

## 16. Phase 6 — Move the canonical local checkout

### Quiescence gate

Before moving the directory:

- stop all subagents and source-editing sessions;
- stop IDE build tasks, file watchers, servers, profilers, compilers, and test processes;
- ensure no second worktree exists;
- require a clean tracked worktree and final remote URL;
- record the artifact stat again;
- require `/home/neroued/ninfer`, `/home/neroued/ninfer-build-pre-cutover`, and
  `/home/neroued/ninfer-eval-venv-pre-cutover` not to exist;
- require `/home/neroued/ninfer-rename-build-pre-cutover` not to exist;
- require `/home/neroued/ninfer-baseline-build-pre-cutover` not to exist.

The current Codex/editor workspace must be closed before the move. An independent parent-directory
shell performs the move, and a fresh workspace/session reopens from the new path; the old workspace
session must not continue editing after its root has moved. The naming goal remains active across
this operational handoff.

### Preserve derived state without trusting it

`build/` and `eval/.venv/` contain old absolute paths. Move them outside the checkout as temporary
recovery material rather than patching or incrementally reusing them:

```bash
(
  set -euo pipefail
  cd /home/neroued
  test ! -e ninfer
  test ! -e ninfer-build-pre-cutover
  test ! -e ninfer-baseline-build-pre-cutover
  test ! -e ninfer-rename-build-pre-cutover
  test ! -e ninfer-eval-venv-pre-cutover
  test "$(stat -c '%d' qwen3.6-ultraspeed)" = "$(stat -c '%d' .)"

  mv -T -- qwen3.6-ultraspeed/build ninfer-build-pre-cutover
  test ! -e qwen3.6-ultraspeed/build && test -d ninfer-build-pre-cutover
  mv -T -- qwen3.6-ultraspeed/cmake-build-qus-baseline \
    ninfer-baseline-build-pre-cutover
  test ! -e qwen3.6-ultraspeed/cmake-build-qus-baseline \
    && test -d ninfer-baseline-build-pre-cutover
  mv -T -- qwen3.6-ultraspeed/cmake-build-ninfer-rename ninfer-rename-build-pre-cutover
  test ! -e qwen3.6-ultraspeed/cmake-build-ninfer-rename \
    && test -d ninfer-rename-build-pre-cutover
  mv -T -- qwen3.6-ultraspeed/eval/.venv ninfer-eval-venv-pre-cutover
  test ! -e qwen3.6-ultraspeed/eval/.venv && test -d ninfer-eval-venv-pre-cutover
  mv -T -- qwen3.6-ultraspeed ninfer
  test ! -e qwen3.6-ultraspeed && test -d ninfer
)
```

The moves are intentionally separate and observable. If any derived-state move fails, reverse only
the already completed moves before retrying. If the directory move completes but the new workspace
cannot open, restore the exact old layout from the parent directory:

```bash
(
  set -euo pipefail
  cd /home/neroued
  mv -T -- ninfer qwen3.6-ultraspeed
  mv -T -- ninfer-build-pre-cutover qwen3.6-ultraspeed/build
  mv -T -- ninfer-baseline-build-pre-cutover qwen3.6-ultraspeed/cmake-build-qus-baseline
  mv -T -- ninfer-rename-build-pre-cutover qwen3.6-ultraspeed/cmake-build-ninfer-rename
  mv -T -- ninfer-eval-venv-pre-cutover qwen3.6-ultraspeed/eval/.venv
)
```

Do not move or copy `out/`, `profiles/`, evaluation runs, `.git/`, or other ignored evidence out of
the checkout; the directory move preserves them in place. Do not create an old-path symlink.

### New-path environment

Reopen at `/home/neroued/ninfer`, then create path-clean derived state:

```bash
(
set -euo pipefail
cd /home/neroued/ninfer
/home/neroued/miniconda3/envs/py311/bin/python -m venv eval/.venv
eval/.venv/bin/python -m pip install \
  -r /home/neroued/backups/ninfer-pre-cutover-eval-requirements.txt
eval/.venv/bin/python -m pip freeze --all \
  > /home/neroued/backups/ninfer-post-cutover-eval-requirements.txt
diff -u /home/neroued/backups/ninfer-pre-cutover-eval-requirements.txt \
        /home/neroued/backups/ninfer-post-cutover-eval-requirements.txt
eval/.venv/bin/python -m pip check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
)
```

The virtual environment reinstall is in scope only because moving it leaves absolute shebangs and
entry points referring to the retired path. The captured full freeze, not the top-level requirements
file alone, preserves transitive dependency versions during recreation.

## 17. Phase 7 — New-path final gate

### Repository and artifact identity

```bash
cd /home/neroued/ninfer
test "$(pwd -P)" = /home/neroued/ninfer
test ! -e /home/neroued/qwen3.6-ultraspeed
test "$(git remote get-url origin)" = https://github.com/Neroued/ninfer.git
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
git status --short --branch

WEIGHTS=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
stat -c '%d:%i:%s' "$WEIGHTS" \
  > /home/neroued/backups/ninfer-post-cutover-artifact.stat
sha256sum -c /home/neroued/backups/ninfer-pre-cutover-artifact.sha256
diff -u /home/neroued/backups/ninfer-pre-cutover-artifact.stat \
        /home/neroued/backups/ninfer-post-cutover-artifact.stat
```

### Build, test, tooling, and smoke

```bash
cmake --build build -j
test "$(ctest --test-dir build -N | awk '/Total Tests:/ {print $3}')" = 54
ctest --test-dir build --output-on-failure \
  -E '^ninfer_engine_mtp_e2e_test$'
(
  set -euo pipefail
  if ctest --test-dir build --output-on-failure \
      -R '^ninfer_engine_mtp_e2e_test$' 2>&1 \
      | tee /home/neroued/backups/ninfer-new-path-known-mtp-failure.log; then
    echo 'known baseline MTP test unexpectedly passed' >&2
    exit 1
  fi
  rg -q 'partial reuse turn 2 parity differs \(mtp-off\)' \
    /home/neroued/backups/ninfer-new-path-known-mtp-failure.log
  rg -q 'partial reuse turn 2 parity differs \(mtp-on\)' \
    /home/neroued/backups/ninfer-new-path-known-mtp-failure.log
)

PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
PYTHONPATH=eval "$PYTHON" -m py_compile \
  $(rg --files eval/ninfer_eval -g '*.py')
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'

"$PYTHON" -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus --quick
./build/src/ninfer \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --prompt '用一句话解释 prefill。' --no-thinking --greedy --max-new 16 \
  > /home/neroued/backups/ninfer-new-path-greedy.txt
diff -u /home/neroued/backups/ninfer-pre-cutover-greedy.txt \
        /home/neroued/backups/ninfer-new-path-greedy.txt

./build/src/ninfer --help
./build/src/ninfer-serve --help
./build/bench/ninfer_bench --help
git diff --check
git status --short --branch
```

Repeat the active Markdown-link and stale-reference audit from Phase 3 at the new path. Search local
ignored configuration for the retired absolute path and either rebuild the derived owner or remove
the stale cache; do not patch generated CMake or virtual-environment files in place.

### Cleanup deferred until final review

Keep all three temporary old build directories and the temporary virtual-environment directory
through the independent Phase 8 review. After its findings are resolved and before the archival
commit, remove those four recovery directories. They are recovery material, not supported
compatibility paths. Retain the Git bundle and baseline evidence until the complete naming goal is
closed.

## 18. Rollback and recovery

Rollback is coordinated at the last completed boundary:

### Before tracked changes merge

- do not merge the branch;
- fix or abandon the branch while `master`, GitHub, and the local path remain unchanged.

### After tracked merge, before GitHub rename

- revert the three naming commits as one coordinated rollback;
- rebuild from a clean directory;
- do not add compatibility aliases to make a partial rename appear usable.

### After GitHub rename, before local move

- prefer a forward fix for documentation, badge, or remote-reference errors;
- for a repository-identity or Git-access failure, renaming the same repository object back to
  `qwen3.6-ultraspeed` and restoring the remote is only an external stabilization step; it does not
  by itself restore the pre-cutover project identity;
- a complete rollback must also revert the three coordinated naming commits on `master`, push the
  reverted result through the restored old origin, rebuild and test from the old source identity,
  and require the old remote and local HEAD to agree;
- never delete the repository or create a second replacement repository.

### After local move

- before new derived state exists, stop all processes and use the immediate inverse moves from
  Phase 6;
- after a new `build/` and `eval/.venv/` exist, first move them to unused
  `/home/neroued/ninfer-post-cutover-build` and
  `/home/neroued/ninfer-post-cutover-eval-venv`, then move the repository root back and restore the
  pre-cutover build trees and virtual environment; do not overwrite either generation;
- if the pre-cutover derived-state backups were already removed, regenerate the required old-path
  state after reverting rather than claiming an exact inverse restore;
- for a complete rollback, restore the GitHub slug and remote, revert and push all three naming
  commits, regenerate path-sensitive state as needed, and rerun the old-path build/test/artifact
  gate;
- do not leave an old GitHub slug with NInfer source or a NInfer local path with reverted QUS source
  as the claimed rollback endpoint.

### Severe local recovery

- restore Git refs from `/home/neroued/backups/ninfer-pre-cutover.bundle`;
- use the recorded refs, repository IDs, artifact hash, and deterministic output to distinguish
  identity problems from data loss;
- do not restore ignored build output as source truth.

## 19. Definition of done

The naming cutover is complete only when all of the following are true:

- the display name, GitHub slug, canonical local path, source namespace/include root, build targets,
  binaries, Python package, and maintained project-owned identifiers use NInfer;
- no QUS aliases, redirects implemented in source, duplicate targets, fallback environment variables,
  or old executable symlinks remain;
- active project documentation reflects NInfer's selected-checkpoint, selected-GPU positioning;
- active documentation accurately separates the current Qwen3.6-27B/RTX 5090 implementation from
  accepted but unimplemented multi-target engine and `.ninfer` container work;
- q5090 v4.2 `.qus` files are byte-identical, validate successfully, and load through the renamed
  executable;
- deterministic real-artifact output matches the pre-cutover baseline;
- a clean build, all 53 baseline-passing CTest entries, Python compile/tests, schema/report checks,
  documentation audit, and CLI/serve/benchmark smoke pass from `/home/neroued/ninfer`, while the
  one known MTP entry reproduces the same two recorded parity-failure markers;
- GitHub repository ID, node ID, default branch, HEAD, and relevant settings survive the in-place
  rename;
- after the archival commit, `origin` uses the new URL, local `master` equals `origin/master`, and
  the worktree is clean;
- the old GitHub slug is documented as reserved and the old local path does not exist;
- temporary derived-state backups have been removed after verification;
- a separate final review confirms that only project identity changed.

## 20. Phase 8 — Final review and archival

This is a broad API, CLI, schema, repository-identity, and local-environment change even though it
does not alter inference algorithms. A separate final review must re-read the complete diff and
evidence after the new-path gate, focusing on:

- accidental algorithmic or numerical edits hidden by mechanical replacement;
- `.qus`/`.ninfer` confusion;
- producer/consumer schema mismatches;
- active documentation claiming unimplemented targets or formats;
- stale absolute paths and mixed QUS/NInfer identities;
- missing GitHub settings or a mismatched repository identity;
- retained aliases or temporary compatibility state.

After all findings are resolved, record the reviewed cutover SHA before the archival commit, then
move this plan and its completion evidence to:

```text
docs/archive/ninfer-foundation/2026-07-14-ninfer-naming-cutover.md
```

Add `ninfer-foundation/` to `docs/archive/README.md` in the same finishing change. The archived plan
must record the reviewed cutover SHA, verification commands and results, GitHub repository identity
check, local artifact identity check, any deviations from this plan, and the disposition of
temporary backup material. It must not claim to record the SHA of the archival commit that contains
the record itself.

Create `docs/archive/ninfer-foundation/README.md` as the era index, update the root archive index,
repeat the active-link, stale-reference, and `git diff --check` audits, then finish through the new
origin:

```bash
(
set -euo pipefail
cd /home/neroued/ninfer
REVIEWED_CUTOVER_SHA=$(git rev-parse HEAD)
mkdir -p docs/archive/ninfer-foundation
git mv docs/plans/2026-07-14-ninfer-naming-cutover.md \
       docs/archive/ninfer-foundation/2026-07-14-ninfer-naming-cutover.md

# Record REVIEWED_CUTOVER_SHA and the completed evidence in the archived plan,
# create docs/archive/ninfer-foundation/README.md, and update docs/archive/README.md.

git diff --check
git status --short
git add docs/archive docs/plans
git commit -m "docs(plan): archive ninfer naming cutover"
git push origin master
git fetch --prune origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
git status --short --branch
test -z "$(git status --porcelain)"
)
```

The final documentation audit uses the same active-file commands as Phase 3 after the move. The new
archive files are checked through archive navigation and direct review, but their preserved
historical commands are not treated as active stale references.
