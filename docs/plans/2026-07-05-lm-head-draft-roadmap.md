# Roadmap: frequency-ranked draft LM head (lm_head_draft)

This is a **milestone roadmap**, not an execution plan. Each milestone states a
goal and defines what a *correct deliverable* looks like. It intentionally does
not prescribe how to implement anything, which files to touch, or task ordering
inside a milestone.

## Purpose

Give the MTP draft path its own small **frequency-ranked LM head**: a shortlist
`V_d ⊂ V` of size `N ≪ V = 248320`, a head `lm_head_draft.weight [N, 5120]`, and
an index map `lm_head_draft.ids [N]`. The draft sites project onto `V_d` (instead
of the full 993 MB Q6 `lm_head`), argmax over `N`, and map the index back to a
real vocab id. Target verify keeps the full head.

## Invariants (must hold across every milestone)

- **Lossless output.** Emitted tokens stay bit-identical to today. The shortlist
  only changes what the draft *proposes*; verify decides what is emitted. This is
  independent of `N` and of the draft head's quantization.
- **Full head retained for all emitting paths.** Only the three MTP *draft*
  projections use `lm_head_draft`; verify, prefill sampling, and the k=0 fallback
  keep the full head.
- **Acceptance is the only quality knob**, and it has up to **two** loss sources:
  1. *Coverage* — the target/draft token falls outside `V_d`. Always present;
     controlled by `N` and by which tokens are in `V_d`. Target: keep per-position
     accepted-token coverage `γ ≥ 0.99` (loss compounds as powers of `γ`).
  2. *Quantization top-1 drift* — a token inside `V_d` is no longer the draft's
     argmax because the draft head's numeric format differs from `lm_head`.
     Present **only** if the draft head is not a value-exact row copy of
     `lm_head`. The default choice (below) makes this term zero.

## Format-invariant note (feature-level)

This feature deliberately revises two stated q5090 v3 invariants and must be
treated as a format contract change, not a minor append:
- "Single resident copy / no derived second copy" — `lm_head_draft` is a second,
  `V_d`-restricted head derived from `lm_head`; it must be defined as a
  first-class weight with its own identity, and the invariant revised to permit
  it explicitly.
- "Value preservation / never a requantization" — a value-exact Q6 row copy keeps
  this invariant intact; any lossy draft-head qtype (e.g. Q4) is a requantization
  and would revise it. The default avoids this.

## Non-goals

- No serve-time monitoring/auto-rebuild; measurement is a one-time offline job on
  a dedicated branch with temporary instrumentation.
- No sampling changes, no tree/parallel drafting, no draft-transformer changes.

## Milestone dependency

M1 → M2 → M3 → M4 → M5 → M6.

---

## M1 — Build the measurement corpus

**Goal.** Assemble a general, mixture-balanced corpus that represents the engine's
intended broad usage, and drive measurement in the way the engine is actually
used, split into a fit set and a per-domain held-out set.

**Deliverable.** A prompt/text set plus its tokenized form with a provenance
manifest (domain/language mixture, per-slice counts, tokenizer identity, source
hashes).

**Correct output means:**
- Tokenized with the model's *own* tokenizer; ids decode back to the source.
- The measurement reflects **real serving form**: because the primary CLI renders
  the Qwen chat template and generates assistant-style text, the corpus drives
  chat-rendered prompts and assistant generation (so role/structural tokens and
  the assistant output distribution are represented). If instead the target is a
  raw-token engine with no chat template, that narrower scope is stated
  explicitly in the manifest.
- The realized mixture matches the intended general weighting across
  language/domain slices with no strong single-domain bias.
- Enough volume that the top-`N` ranking is stable up to ~64k.
- Train and per-domain held-out splits are disjoint; fully reproducible from the
  manifest.

---

## M2 — Dedicated branch: temporary statistics support

**Goal.** On a new branch, add temporary instrumentation that measures both (a)
the distribution the draft must cover and (b) enough per-round detail to predict
acceptance under any candidate shortlist.

**Deliverable.** A build that consumes M1 and emits, per split:
- a **ranking distribution** over vocab (which tokens the model actually emits),
  measured degeneration-free (e.g. teacher-forced next-token argmax over realistic
  text, not free-running greedy loops); and
- a **per-round acceptance record** from the real MTP decode path: for each draft
  position, the target argmax token, the full-head draft argmax token, and the
  accepted-prefix length — sufficient to *replay* acceptance for an arbitrary
  `V_d` offline.

**Correct output means:**
- The ranking distribution counts what the model emits (not raw input unigrams),
  and is not biased by generation degeneration.
- The acceptance record is exact and complete enough that, given any `V_d`, the
  resulting accepted-prefix length per round is reconstructable without re-running
  the model (a token is "still accepted under `V_d`" iff it was accepted with the
  full head **and** the full-head draft token is in `V_d`).
- Instrumentation does not change normal inference results, is clearly isolated as
  temporary/branch-only, memory-safe (sanitizer-clean), and deterministic for a
  fixed corpus.

---

## M3 — Produce the frequency and acceptance data

**Goal.** Run the M2 instrumentation over M1 and produce the artifacts needed for
shortlist selection and exact acceptance prediction.

**Deliverable.** Per-split ranking distributions and per-round acceptance-replay
datasets, plus a coverage/acceptance summary.

**Correct output means:**
- The ranked distribution is stable (top-`N` membership robust to resampling;
  train/held-out agree on the high-frequency head) and its head is dominated by
  expected structural tokens (sanity check).
- The acceptance-replay dataset reproduces the *baseline* measured acceptance
  (e.g. `accepted_per_pos`, acceptance length) exactly when replayed with
  `V_d = V`, proving it is faithful.
- Both artifacts are schema-documented and reproducible from M1 + M2.

---

## M4 — Turn data into a draft head

**Goal.** Choose the shortlist `V_d` and the head qtype, and materialize the draft
head content, with an acceptance/tok-s justification grounded in exact replay.

**Deliverable.** The shortlist id list, the draft-head weight content (the
`lm_head` rows for `V_d`, in the chosen qtype), and a selection report:
per-domain `γ(N)`, predicted accepted-per-round and decode tok/s, and — if a lossy
qtype is chosen — the measured additional top-1 drift.

**Correct output means:**
- `V_d` contains all force-include tokens (special/control tokens and the
  byte-fallback tokens) plus the highest-ranked remaining tokens up to `N`.
- The id list is valid, unique, in range; the id → vocab mapping is exact.
- **Default head qtype is a value-exact Q6 row copy** of `lm_head`'s rows for
  `V_d` (dequantized draft rows equal dequantized `lm_head` rows byte-for-byte →
  zero quantization drift → acceptance loss is pure coverage). A lower-precision
  variant is allowed only if its additional top-1 drift is measured and reported.
- Acceptance prediction comes from **offline shortlist replay** on the M3 records
  (not a single global-γ assumption); worst-domain held-out coverage meets the
  target (`γ(N) ≥ 0.99`).
- The chosen `N`/qtype is justified as the predicted tok/s optimum under the
  coverage constraint.

---

## M5 — Revise the quantization format to carry the draft head

**Goal.** Represent the draft head as a first-class part of the packed format,
including the deliberate revision of the invariants it touches, so one artifact
carries everything the runtime needs.

**Deliverable.** An updated format + converter that emits a `.qus` containing the
draft-head weight block and the id-map block, with updated format documentation
that records the revised invariants and the new contracts.

**Correct output means:**
- The format change is explicit and consistent across the plan/converter, the
  loader/verifier, and the format doc: the revised single-copy invariant, a new
  `LM_HEAD_DRAFT` (and id-map) source kind, and a new contract for a
  `CONTIGUOUS` integer (id-map) block — today `CONTIGUOUS` only covers BF16/FP32
  control tensors.
- The MTP module's structural expectations are updated deliberately (its current
  hard-checked tensor/segment/fusion counts must be revised to the new totals,
  not silently broken).
- The produced `.qus` passes the (updated) verifier; its manifest lists both new
  blocks with correct shape/qtype/module; reading them back yields exactly the M4
  content (bit-exact ids; packed weight equal to M4's rows).
- Malformed/inconsistent draft blocks are rejected rather than loaded silently;
  the rest of the model still loads.

---

## M6 — Engine-side draft-head support

**Goal.** Make the runtime use the draft head at the MTP draft sites and validate
the whole feature end to end, including evidence that the full-head draft cost is
actually removed.

**Deliverable.** A runtime that loads and uses `lm_head_draft` for drafting, plus
a validation report with the chosen final `N`.

**Correct output means:**
- **Output tokens are bit-identical** to a full-head-draft run on the same inputs
  (the lossless gate).
- **Cost-removal evidence:** a profile/trace or report shows the MTP draft sites
  read only `lm_head_draft [N,5120]` (draft-head bytes/time scale ~ `N/V`, not the
  full 993 MB), produce `[N,1]` draft logits, and correctly map the shortlist
  index back to a vocab id; the full head is read only in verify/prefill/fallback.
- The decode round remains graph-capturable and sanitizer-clean.
- Measured acceptance matches the M4 replay prediction within tolerance, and
  decode tok/s improves over the current baseline; the final `N`/qtype is
  confirmed (or re-selected) from measured results.

---

## Definition of done (whole effort)

A committed `.qus` with a validated `lm_head_draft`, a runtime that uses it with
bit-identical output, profiler-confirmed removal of the redundant full-head draft
reads, and measurably higher decode tok/s. A short report captures the corpus
mixture and serving form, the coverage curve, and predicted-vs-measured
acceptance from replay. The temporary M2 instrumentation may be dropped once M6
is validated.
