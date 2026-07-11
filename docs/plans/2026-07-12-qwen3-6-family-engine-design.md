# Qwen3.6 Family Engine Elevation Design And Implementation Plan

> Status: proposed on 2026-07-12; implementation has not started.
>
> This document defines how the project will move from one fixed Qwen3.6-27B deployment to an
> explicitly supported Qwen3.6 family while preserving its one-RTX-5090, one-user, static-schedule
> specialization. Until the corresponding implementation phase lands, current active documents
> remain authoritative: [`../design.md`](../design.md) and
> [`../qwen3.6-27b-architecture.md`](../qwen3.6-27b-architecture.md) describe the implemented model
> scope until the family cutover is complete, while
> [`../q5090_packed_file_format_v4.md`](../q5090_packed_file_format_v4.md) remains the artifact
> authority only until the atomic v5 cutover in Phase 2.

## 1. Decision

The project will support a finite, compiled registry of exact Qwen3.6 model profiles rather than
becoming a configuration-driven general-purpose runtime.

The initial registry will contain:

- the existing dense Qwen3.6-27B profile;
- Qwen3.6-35B-A3B, a sparse MoE profile whose local BF16 source is
  `/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16`.

An artifact will identify one exact model profile and one exact packing profile. The Engine will
select one concrete implementation after CPU-side artifact parsing and schema validation. Inside
that implementation, model dimensions, weight structures, layer order, and performance-critical
dispatch will remain compile-time or otherwise fixed at load time. There will be no per-layer
virtual dispatch, name lookup, filesystem access, runtime graph construction, or weight repacking on
the inference hot path.

This changes the project's proposition from:

> one engine for one Qwen3.6-27B checkpoint shape

to:

> one RTX-5090-specialized engine for an explicit set of qualified Qwen3.6 family profiles

It does not change the single-user, single-active-sequence, single-GPU deployment target.

### 1.1 Purpose and expected value

This elevation changes the unit of support. Today the checkpoint, model schedule, artifact format,
runtime memory formulas, tools, and product identity are effectively one indivisible 27B object.
That maximizes initial optimization speed, but a second architecture would otherwise force either a
forked engine or a gradual spread of model-name conditionals and duplicated constants.

The family design creates one deliberate extension seam at each real boundary:

- an exact model profile defines mathematics and state geometry;
- an exact packing profile defines how that model is represented for the RTX 5090;
- one load-time Engine selection preserves static, separately tunable hot paths;
- shared L0/L1 facilities are reused only where their contracts are truly model-independent;
- frontends and evidence always derive identity from the loaded artifact.

Qwen3.6-35B-A3B is a useful forcing function because it changes hidden width, layer count, GQA/GDN
ratios, FFN topology, MTP topology, and Vision output width at once. Supporting it correctly proves
that the engine's reusable boundaries are real rather than a renamed collection of 27B constants.
The intended long-term value is therefore not a model zoo: it is the ability to qualify another
explicit Qwen3.6 profile without redesigning storage, lifetime, reporting, and serving ownership each
time, while preserving profile-specific CUDA performance.

## 2. Goal

Add Qwen3.6-35B-A3B as a first-class Text, MTP, Vision, CLI, OpenAI, Anthropic, conversion,
reference, and benchmark target while preserving the existing 27B behavior and optimization model.

The completed system must:

- identify the loaded model from the artifact instead of a caller-selected model-type flag;
- reject every artifact that does not exactly match a compiled model and packing profile;
- retain straight-line, shape-specialized Text/MTP/Vision execution inside each profile;
- express dense and MoE FFNs as two explicit L2 model structures;
- store all routed experts in a direct-consumption q5090 layout with no runtime repack;
- execute routing and expert selection entirely on device, including CUDA Graph replay;
- provide dedicated decode, MTP/small-token, and prefill MoE paths;
- derive KV, GDN, step-buffer, workspace, and Vision memory from the selected profile;
- carry checkpoint-appropriate tokenizer, chat-template, processor, and model identity metadata;
- report model profile, packing profile, and artifact identity in serving and benchmark output;
- keep the existing external OpenAI and Anthropic protocol contracts coherent;
- qualify both profiles independently for numerical correctness, memory safety, observable behavior,
  and caller-visible performance.

## 3. Non-Goals

This work will not:

- accept arbitrary Hugging Face `config.json` files;
- support arbitrary Qwen3.5/Qwen3.6 variants merely because some dimensions happen to match;
- introduce a dynamic model graph, operator registry keyed by model strings, or per-layer virtual
  dispatch;
- add batching, concurrent GPU sequences, multi-GPU execution, expert parallelism, tensor
  parallelism, or CPU expert offload;
- page experts from storage or host memory on demand;
- add runtime weight conversion, dequantized resident copies, or alternate weight loaders;
- preserve q5090 v4.2 compatibility after the v5 migration is complete;
- infer model semantics solely from tensor names, marketing names, or dimension coincidences;
- promise the model card's extended million-token operating point without separate RoPE, memory,
  numerical, and performance qualification;
- claim that a 3B active-parameter model needs only 3B parameters resident;
- make the runtime execute arbitrary Jinja chat templates or arbitrary processor configuration.

## 4. Current Facts

### 4.1 Existing system shape

The implemented engine is intentionally fixed to Qwen3.6-27B:

- one `ModelConfig` owns all Text, GDN, attention, MTP, and derived dimensions;
- `Qwen3_6_27B` owns the dense Text schedule and dense MTP schedule;
- `Qwen3_6_Vision` owns a fixed 27-layer tower whose merger emits width 5120;
- `Engine` duplicates the 27B geometry in cache, state, step-buffer, and workspace formulas;
- q5090 v4.2 combines a binary container, one exact model card, and one exact packing policy;
- the converter, verifier, Python reference, real-artifact tests, benchmark shapes, and serving
  default model id are all 27B-specific.

The current fixed dimensions are visible in [`../../include/qus/model/config.h`](../../include/qus/model/config.h).
The parser then requires those same values exactly in
[`../../src/core/weight_store_parser.cpp`](../../src/core/weight_store_parser.cpp), and the converter
has a second fixed copy in [`../../tools/q5090_convert/convert.py`](../../tools/q5090_convert/convert.py).

These checks are valuable: they prevent a structurally valid artifact from being interpreted with
the wrong model schedule. The family design must replace them with exact profile validation rather
than simply removing them.

### 4.2 Profile comparison

The local BF16 checkpoints establish the following model facts:

| Field | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| HF architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |
| HF model type | `qwen3_5` | `qwen3_5_moe` |
| Text hidden | 5120 | 2048 |
| decoder layers | 64 | 40 |
| GDN / full-attention layers | 48 / 16 | 30 / 10 |
| full-attention Q / KV heads | 24 / 4 | 16 / 2 |
| attention head dimension | 256 | 256 |
| Q heads per KV head | 6 | 8 |
| GDN QK / V heads | 16 / 48 | 16 / 32 |
| GDN head dimension | 128 | 128 |
| GDN V heads per QK head | 3 | 2 |
| GDN convolution width | 4 | 4 |
| dense FFN intermediate | 17408 | absent |
| routed experts / top-k | absent | 256 / 8 |
| expert intermediate | absent | 512 |
| shared expert intermediate | absent | 512 |
| vocabulary | 248320 | 248320 |
| MTP layers | 1 dense | 1 MoE |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 | 27 / 1152 / 4304 |
| Vision merger output | 5120 | 2048 |

Both profiles retain:

- three GDN layers followed by one full-attention layer in each four-layer group;
- attention head dimension 256 and partial rotary dimension 64;
- RoPE theta `1e7`, interleaved MRoPE sections `[11,11,10]`, and RMS epsilon `1e-6`;
- GDN key/value head dimension 128, width-4 causal convolution, and FP32 recurrent state;
- gated full-attention output, zero-centered ordinary/QK RMSNorm, and plain GDN gated RMSNorm;
- an untied embedding and full `lm_head`;
- the same Vision backbone geometry, but not the same Vision weights.

The explicit 35B facts come from
[`config.json`](</home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16/config.json>) and the
safetensors index under the same directory. The marketing name is not a sufficient dispatch key:
both implementations still use upstream `qwen3_5*` architecture identifiers, and the existing 27B
source also contains checkpoint-specific fields. The converter must normalize source metadata into a
project-owned exact profile.

### 4.3 Real 35B tensor inventory

For every 35B Text layer, and for the one MTP layer, the FFN tensors are:

```text
mlp.experts.gate_up_proj                    [256,1024,2048]
mlp.experts.down_proj                       [256,2048,512]
mlp.gate.weight                             [256,2048]
mlp.shared_expert.gate_proj.weight          [512,2048]
mlp.shared_expert.up_proj.weight            [512,2048]
mlp.shared_expert.down_proj.weight          [2048,512]
mlp.shared_expert_gate.weight               [1,2048]
```

`experts.gate_up_proj` is already fused in source order. For each expert, rows `[0,512)` are the
SiLU gate and rows `[512,1024)` are the multiplicative up projection. Expert id `e` is the same
identity as router row `e`; no implicit expert permutation is allowed in the initial format.

Other derived 35B widths are:

```text
attention Q / KV width              4096 / 512
attention raw q_proj rows           8192 (query and output gate interleaved per head)
attention o_proj                    [2048,4096]
GDN Q / K / V / Z width             2048 / 2048 / 4096 / 4096
GDN convolution rows                8192
GDN in_a / in_b                     [32,2048]
GDN out_proj                        [2048,4096]
MTP fc                              [2048,4096]
Vision merger fc2                   [2048,4608]
```

The attention q-projection remains interleaved as `[query_256 | gate_256]` per head. The converter
must continue to perform the existing value-preserving de-interleave transform; a simple split of
the first and second halves is wrong.

### 4.4 Parameter and memory facts

Header-only inspection of the local BF16 files gives:

| Component | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| embedding | 1.271B | 0.509B |
| Text layers | 24.353B | 33.643B |
| final norm + full head | 1.271B | 0.509B |
| Vision | 0.461B | 0.447B |
| MTP | 0.425B | 0.845B |
| total | 27.781B | 35.952B |

The full 35B expert banks must remain resident because any expert can be selected by the next token.
The approximately 3B active count describes work and bytes touched per token, not residency.

At context 262144, payload-only BF16 state is approximately:

| State | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| Text KV | 16 GiB | 5 GiB |
| one-layer MTP KV | 1 GiB | 0.5 GiB |
| one GDN recurrent snapshot slot | 144 MiB | 60 MiB |
| one GDN convolution snapshot slot | 2.81 MiB | 1.41 MiB |

The 35B profile therefore has larger resident weights but materially smaller KV and recurrent state.
The final qualified context remains a measured runtime property, not a model-card inheritance.

### 4.5 Tokenizer, template, and processor facts

The two local checkpoints have the same base BPE vocabulary mapping and the same core multimodal and
thinking token ids, but their embedded asset bytes and chat templates are not identical. The 35B
source template is
[`chat_template.jinja`](</home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16/chat_template.jinja>).

The existing q5090 artifact already embeds checkpoint-specific tokenizer assets, which is the right
ownership. Family support must not replace that with one global tokenizer assumption. Chat-template
and processor behavior also need explicit profile identity even if the two initial processor
geometries happen to match.

## 5. Design Principles

The implementation follows these principles:

1. **Exact profiles, not arbitrary configs.** Source configs are normalized and then matched to a
   compiled allowlist.
2. **One load-time choice.** Dynamic selection stops at the concrete Engine implementation boundary.
3. **Static hot paths.** Concrete profiles retain fixed arrays, straight-line layer order, and finite
   shape dispatch.
4. **Semantic L1 APIs.** Public operators express routing or expert mathematics, not one CUDA launch
   strategy.
5. **No hidden residency.** All persistent allocations and all workspace requirements are planned
   before execution.
6. **Direct artifact consumption.** Quantized weights stay in their canonical q5090 representation
   through decode and prefill.
7. **Structural and semantic validation remain separate.** A safe container is not automatically a
   valid model.
8. **27B remains a first-class optimized profile.** Family support is not permission to replace tuned
   27B kernels with slower generic kernels.
9. **Every claim names its profile.** Artifacts, reports, server model ids, parity results, and
   performance evidence are never ambiguous between 27B and 35B-A3B.

## 6. Target Engine Architecture

### 6.1 Profile model

Introduce a project-owned profile definition with an enum or numeric identity, a compile-time spec,
and a read-only runtime view:

```cpp
enum class ModelProfileId {
    Qwen36Dense27B,
    Qwen36Moe35BA3B,
};

struct Qwen36Dense27BSpec;
struct Qwen36Moe35BA3BSpec;

struct ModelInfo;  // immutable runtime mirror for reporting/frontends
```

The initial stable semantic identities are distinct from display/serving names:

| Meaning | 27B | 35B-A3B |
|---|---|---|
| family id | `qwen3_6` | `qwen3_6` |
| model profile id | `qwen3_6_dense_27b_v1` | `qwen3_6_moe_35b_a3b_v1` |
| canonical serving id | `qwen3.6-27b` | `qwen3.6-35b-a3b` |

Numeric ABI values and the final packing-profile ids are frozen in the v5 specification. A packing
profile identifies storage policy, not model mathematics, and cannot substitute for the model
profile id.

Each spec owns the fields listed in the Goal and additionally provides exact layer-index maps:

- `layer_kind[layer]`;
- `full_slot[layer]` or `NO_SLOT`;
- `gdn_slot[layer]` or `NO_SLOT`;
- `ffn_kind[layer]`;
- the independent MTP layer profile.

The slot maps are derived once from the exact layer table. KV and GDN code must not keep formulas
that assume every fourth layer forever.

The runtime `ModelInfo` is generated from the same profile definition; it is not an independently
maintained second table.

### 6.2 Load sequence

The new load flow is:

```text
open stable artifact fd
  -> parse bounded v5 header/catalog/tokenizer/model metadata without CUDA
  -> validate container structure and arithmetic
  -> resolve model_profile_id and packing_profile_id in compiled registry
  -> validate normalized model record and exact tensor inventory
  -> resolve requested optional modules
  -> build exact weight/cache/workspace residency plan
  -> initialize CUDA
  -> upload selected module payloads
  -> construct concrete EngineImpl and bind typed weight tables
  -> initialize stable step/sampling/routing buffers
  -> allow prefill/decode/serve calls
```

Artifact identity selects the implementation. A command-line `--model-type` must not be required and
must not be able to override the artifact.

### 6.3 Engine type erasure

The public `Engine` retains one stable API and owns one concrete loaded implementation through a
whole-engine variant or type-erased pointer:

```text
Engine
  `-- EngineImpl<Qwen36Dense27BSpec>
  `-- EngineImpl<Qwen36Moe35BA3BSpec>
```

Type erasure occurs around whole calls such as `prefill`, `decode_step`, memory statistics, and MTP
statistics. It must not occur around individual layers or operators. Concrete EngineImpl storage
must not move after CUDA Graph capture.

### 6.4 L2 Text structure

The shared Text skeleton is:

```text
embedding
for exact layer in profile:
  h = zero-centered RMSNorm(x)
  x = x + FullAttention(h) or GDN(h)
  h = zero-centered RMSNorm(x)
  x = x + DenseSwiGLU(h) or QwenSparseMoE(h)
final zero-centered RMSNorm
full lm_head and sampling
```

Use explicit weight structures:

```cpp
struct DenseFfnW;
struct MoeFfnW;

template<class FfnW> struct FullLayerW;
template<class FfnW> struct GdnLayerW;
template<class FfnW> struct MtpW;
```

Templates are an implementation option, not a requirement. Two concrete cards sharing common
attention/GDN helpers are acceptable if that is clearer. The required property is that a dense layer
does not carry a runtime MoE branch and a MoE layer does not pretend to be three ordinary dense
weights.

### 6.5 MTP structure

MTP shares the stem and full-attention semantics across both profiles:

```text
normalized token embedding + normalized target hidden
  -> fc(concat)
  -> one full-attention decoder layer
  -> profile-specific dense or MoE FFN
  -> final norm
```

The 35B MTP source contains its own router, 256 expert banks, shared expert, and shared gate. Target
Text and MTP must therefore use the same MoE mathematical contract but distinct weights and caches.

The proposal head remains separate from target correctness. A 35B shortlisted head must be regenerated
with width 2048; the current width-5120 draft block cannot be reused.

### 6.6 Vision and multimodal processing

The two initial profiles share the Vision backbone geometry, so its block implementation can be
reused. The Vision profile still declares every dimension explicitly and statically checks the shared
implementation assumptions.

The merger output, workspace output tensor, scatter validation, and Text placeholder injection use
`vision.out_hidden == text.hidden`. They must not contain a literal 5120.

The processor receives a `MultimodalProfile` containing patch/merge geometry, normalization,
position behavior, and special token identities. Media safety budgets remain runtime policy, not
model geometry.

### 6.7 Resource planning ownership

Physical arenas remain owned by runtime/L0. Exact size formulas move to the components that know the
layout:

- `KVCache::required_bytes(...)` from the same arguments as its constructor;
- `GdnState::required_bytes(...)` from the same arguments as its constructor;
- profile-specific `StepStateLayout`;
- dense and MoE Text workspace plans by phase and token regime;
- profile-specific Vision workspace planning.

The planner returns an exact allocation sequence as well as a conservative capacity. Tests compare
planned capacity with observable arena high-water behavior; they do not lock private allocation call
order.

## 7. q5090 v5 Artifact Contract

### 7.1 Version decision

Family support requires a new major artifact revision because it changes:

- model metadata interpretation;
- layer topology representation;
- source-role inventory;
- MoE expert-bank layout;
- MTP inventory;
- Vision and draft-head shape interpretation;
- parser and loader descriptors.

The implementation will move directly to q5090 v5, but specification promotion and runtime cutover
must be atomic. During Phase 1, v5 remains a candidate contract owned by this active plan; it must
not be described as the implemented or normative format while the runtime still consumes v4.2.
During the Phase 2 cutover:

- the candidate contract is promoted as `docs/q5090_packed_file_format_v5.md` and replaces the active
  specification in the same change that switches the converter, reader, verifier, C++ parser,
  loader, fixtures, and runtime to v5 only;
- the final active v4.2 specification is preserved as
  `docs/archive/optimization-era/q5090_packed_file_format_v4_2.md`, avoiding collision with the
  earlier archived `q5090_packed_file_format_v4.md` snapshot;
- the real 27B artifact is regenerated as v5 and becomes the canonical runtime input;
- v4 branches, aliases, fallback parsers, and compatibility tests are deleted.

The 35B artifact is then generated from that already-active v5 contract in Phase 4. No supported
revision may leave the active specification describing a format different from the one accepted by
the runtime.

The container name should not continue to encode only `w4g64`: qtype and layout are already
self-describing per block, and the MoE packing mixture is profile-specific.

Proposed canonical artifact names are:

```text
out/qwen3_6_27b.q5090_mixed_v5.qus
out/qwen3_6_35b_a3b.q5090_mixed_v5.qus
```

### 7.2 Separation of records

v5 separates three contracts:

| Contract | Owns |
|---|---|
| container ABI | offsets, sizes, records, alignment, qtypes, layouts, code planes, CRC |
| model profile | exact Text/MTP/Vision mathematics, dimensions, layer table, feature flags |
| packing profile | qtype assignment, fusion policy, shortlist policy, expert physical layout |

A fixed binary `ModelRecord` should include:

- family and model profile ids;
- Text geometry;
- attention and GDN geometry;
- RMS/RoPE/MRoPE parameters and feature flags;
- an FFN union whose inactive fields are required to be zero;
- MTP geometry and FFN kind;
- complete Vision geometry;
- embedding/head tying flags;
- tokenizer, chat-template, processor, and special-token profile ids;
- normalized source-config hash and safetensors-index hash;
- offsets/counts for exact `LayerRecord` entries.

Each `LayerRecord` contains at least mixer kind and FFN kind. The model profile id selects compiled
code; the redundant record fields are still validated exactly to catch corruption and converter
mistakes.

The sidecar manifest mirrors these fields but remains informational. Runtime decisions come only
from the binary.

### 7.3 Module and source-role model

The existing module boundary remains appropriate:

- `TEXT_CORE` includes Text router, experts, and shared experts;
- `MTP_DRAFT` includes MTP router, experts, and shared experts;
- `VISION_ENCODER` remains independent;
- `LM_HEAD_DRAFT` remains optional.

No separate optional MoE module is added. A Text or MTP module without its required expert banks is
invalid.

Presence semantics are explicit:

- `TEXT_CORE` is mandatory and appears exactly once;
- `MTP_DRAFT` and `VISION_ENCODER` are selectively present artifact capabilities; when absent, the
  Engine disables speculative decoding or multimodal requests explicitly rather than synthesizing
  weights or changing model identity;
- `LM_HEAD_DRAFT` is optional, may appear only with `MTP_DRAFT`, and changes proposal quality only;
- both canonical family release artifacts must include Text, MTP, and Vision, even though the
  container permits Text-only diagnostic artifacts.

Profile validation checks a complete inventory for every present module. Serving rejects a request
whose required capability is absent before beginning inference.

Add semantic source roles equivalent to:

```text
MOE_ROUTER
MOE_EXPERT_GATE_UP
MOE_EXPERT_DOWN
MOE_SHARED_GATE
MOE_SHARED_UP
MOE_SHARED_DOWN
MOE_SHARED_GATE_SCORE
```

Shared gate/up may use an ordinary fused block with two segments. Expert gate/up is already one
source tensor and should remain one bank role rather than two cross-expert segments.

### 7.4 Expert-bank layout

Do not create one TensorEntry per expert. That would create more than twenty thousand entries for
the two routed banks in the 40 Text layers alone.

The preferred v5 candidate is `EXPERT_ROW_SPLIT`, with one logical `[E,N,K]` TensorEntry per bank and
one ordinary row-split mini-block per expert:

```text
Kp = align_up(K, 128)
G  = Kp / group_size

nibble_e = N * G * nibble_bytes_per_group
high_e   = N * G * high_bytes_per_group
scale_e  = N * G * sizeof(fp16)

high_rel_e  = align_up(nibble_e, 256)
scale_rel_e = high_rel_e + align_up(high_e, 256)
expert_payload = scale_rel_e + scale_e
expert_stride  = align_up(expert_payload, 256)

expert_base(e) = bank_base + e * expert_stride
bank_payload   = E * expert_stride
```

This provides:

- direct `expert_id -> base + stride` addressing;
- zero-copy ordinary Weight views of one selected expert;
- reuse of existing Q4/Q5/Q6/W8 code-plane decoding inside an expert;
- sequential, bounded converter/verifier processing;
- no runtime repack or expert-id map;
- a bounded catalog with approximately 1024 blocks for the proposed full 35B artifact.

Before the ABI is frozen, a real-shape microbenchmark must compare this bank-local layout with the
simpler alternative of flattening `[E,N,K]` to `[E*N,K]` in global ROW_SPLIT planes. The final choice
is based on selected-expert decode and grouped-prefill evidence, not intuition alone.

Whichever layout wins, v5 forbids:

- one catalog entry per expert;
- runtime expert permutation or repacking;
- any frequency-based or other expert reordering in v5;
- a bank descriptor whose expert identity differs from router row identity.

### 7.5 Initial packing candidate

The initial 35B packing candidate is:

| Tensor class | Candidate qtype/layout |
|---|---|
| routed expert gate/up bank | Q4G64 expert bank |
| routed expert down bank | Q5G64 expert bank |
| shared gate/up | Q4G64 ordinary fused ROW_SPLIT |
| shared down | Q5G64 ROW_SPLIT |
| router | BF16 CONTIGUOUS |
| shared scalar gate | BF16 CONTIGUOUS |
| MTP linears and expert banks | W8G32 initially |
| embedding and full head | Q6G64 |
| optional draft head | Q4G64 shortlist |

Router and shared-gate quantization are not part of the initial scope. Router error can change the
discrete expert set and is more dangerous than ordinary projection error.

Using the real source shapes and this candidate policy gives an estimated packed payload:

| Payload class | Estimated size |
|---|---:|
| 40-layer MoE tensors | 17.29 GiB |
| complete TEXT_CORE | 18.78 GiB |
| VISION_ENCODER | 0.26 GiB |
| LM_HEAD_DRAFT | 0.13 GiB |
| W8 MTP_DRAFT | 0.84 GiB |
| complete artifact payload | 20.02 GiB |

These are deterministic format estimates, not an accepted quality, residency, or throughput claim.
The qtype assignment becomes normative only after quantization parity and end-to-end evaluation.

### 7.6 Converter architecture

Replace the converter's single global `EXPECTED` dictionary with project-owned normalized source
profiles. Source normalization handles checkpoint packaging differences such as nested config
fields, `dtype` versus `torch_dtype`, and upstream architecture names. It does not create new runtime
profiles automatically.

The tensor-plan builder receives one exact profile and one packing profile:

```text
normalized source config
  -> exact profile match
  -> dense or MoE Text plan
  -> dense or MoE MTP plan
  -> profile Vision plan
  -> optional profile-width draft-head plan
  -> v5 catalog and payload writer
```

Expert banks must be processed in bounded slices. The converter must not materialize an entire
multi-hundred-megabyte packed bank or all expanded quantization intermediates at once. With a
bank-local layout it can read one expert slice, quantize it, write its final region, update integrity
state, and release temporary memory.

The writer keeps the existing transactional behavior: bounded temporary output, complete
verification, fsync, and atomic rename. Neither converter nor tests choose artifacts by glob or
modification time.

### 7.7 Structural validation

The L0 parser validates safety independently of model support:

- every offset/size/product uses checked unsigned arithmetic;
- record counts and string bytes have explicit caps;
- model, layer, tokenizer, and tensor regions are adjacent or aligned exactly as specified;
- every tensor dimension is nonzero and representable by runtime descriptors;
- layout, qtype, group size, scale dtype, plane sizes, and payload formulas agree;
- bank `E*N`, expert stride, plane offsets, and full payload size cannot overflow;
- modules and payloads are ordered, bounded, aligned, and non-overlapping;
- segment/fusion ownership is complete and non-overlapping;
- source-layer bounds are checked against the correct Text, MTP, or Vision count;
- unknown ids, flags, layouts, qtypes, and nonzero reserved fields are rejected;
- required metadata/tokenizer padding is zero;
- selected payload integrity is checked according to the final trusted-artifact policy.

CRC detects accidental corruption; it is not an authentication mechanism. Safety cannot depend on an
attacker being unable to recompute CRC.

### 7.8 Exact profile validation

After structural parsing and before CUDA initialization, the selected profile validates:

- the complete ModelRecord and LayerRecord sequence;
- every required module and its exact role inventory;
- no duplicate or missing `(module, role, layer)` identity;
- embed and full head shape `[vocab,H]`;
- dense FFN shapes for the dense profile;
- router `[E,H]`, gate/up bank `[E,2I,H]`, and down bank `[E,H,I]` for every MoE layer;
- shared gate/up/down and scalar-gate shapes;
- general safety requires `0 < top_k <= experts`, and the exact 35B profile additionally requires
  `experts == 256` and `top_k == 8`;
- expert-axis index equal to router row id, with no expert id map or permutation in v5;
- full-attention, GDN, control, norm, and fusion shapes;
- MTP FFN kind and complete inventory;
- Vision merger output equal to Text hidden;
- draft head `K == H`, matching id-map length, and valid unique vocab ids;
- qtype and layout assignment equal to the selected packing profile.

This replaces the parser's current fixed-27B assertion and
`WeightStore::require_mtp_module_expectations()`. It does not weaken them.

### 7.9 Runtime descriptors and manifest

Keep ordinary `Weight` for ordinary matrices and add a direct, non-owning expert-bank descriptor:

```cpp
struct ExpertWeightBank {
    const void* payload;
    QType qtype;
    int experts;
    int n;
    int k;
    size_t expert_stride;
    // per-expert plane offsets and validated metadata
};
```

`expert(e)` may construct a zero-copy temporary Weight view or the MoE operator may consume the bank
directly. No descriptor allocates or repacks device memory.

The manifest continues to mirror the binary and adds structured fields for:

- model family/profile and normalized config;
- packing profile and quantizer contract;
- expert layout;
- module and expert-bank resident bytes;
- active top-k expert bytes per token;
- router/shared bytes;
- tokenizer/template/processor hashes;
- source config and safetensors-index hashes.

A single `effective_text_bpw` is insufficient for MoE because resident bytes and active bytes differ
substantially.

## 8. MoE Mathematical And Operator Contract

### 8.1 Reference semantics

For post-attention normalized token state `h`, the 35B inference contract is:

```text
router_logits = W_router h                       # [E]
router_probs  = softmax(router_logits, FP32)
(values, ids) = topk(router_probs, top_k)
routing_weight = values / sum(values)
routing_weight = cast_to(router_logits.dtype)

for each selected expert e:
  gate, up = split(W_gate_up[e] h, [I,I])
  y_e = W_down[e](SiLU(gate) * up)

y_routed = sum_in_defined_order(routing_weight[e] * y_e)

y_shared = W_shared_down(
    SiLU(W_shared_gate h) * W_shared_up h
)
shared_scale = sigmoid(W_shared_gate_score h)

output = y_routed + shared_scale * y_shared
```

There is no inference capacity factor, token dropping, stochastic routing, or auxiliary-loss term.
The shared expert is not routed expert 256; it always executes and has an independent scalar gate.

The numerical contract must explicitly freeze:

- the BF16 input rounding boundary;
- router projection accumulation and output dtype;
- FP32 softmax;
- top-k tie policy;
- post-top-k normalization and cast point;
- expert activation and projection accumulation precision;
- routed expert reduction order and output rounding;
- shared scalar sigmoid precision;
- final routed/shared addition order.

Although selected-logit softmax can be algebraically equivalent to full-softmax followed by
renormalization, it is not adopted until its real BF16/router parity is demonstrated.

### 8.2 L1 public surface

The public surface describes mathematical work. Two acceptable decompositions are:

```text
topk_softmax_router
routed_swiglu_experts
shared_swiglu_expert
weighted_expert_reduce
```

or one public `sparse_moe` contract with private token-regime launchers. The final choice should make
independent numerical oracles and workspace ownership clear without exposing block size, expert CTA
mapping, histogram implementation, or GEMM strategy.

The model layer owns that every decoder layer invokes this operator. L1 owns decode/small-T/prefill
dispatch, expert-bank addressing, and launch policy.

### 8.3 One-token decode

For `T == 1`, every layer selects exactly eight routed experts and one shared expert. The path must:

- compute router logits, top-k ids, and weights on device;
- keep ids and weights in stable Engine-owned or workspace addresses;
- launch a fixed topology independent of expert identity;
- process the eight selected expert gate/up matrices with sufficient combined parallelism;
- process expert down projections and weighted reduction without host intervention;
- execute or fuse the shared branch where numerically justified;
- consume expert-bank bytes directly.

It must not read expert ids back to the CPU or issue 24 host-selected projection launches per layer.
A likely private mapping is CTA work over `(route_slot, output_row_tile)`, but that is a kernel
decision rather than an API contract.

### 8.4 MTP and small-T

Autoregressive MTP proposal and target candidate-window verification normally use `T` between one
and six. This regime needs a separate selected-expert path rather than being forced through either
the scalar decoder or a large prefill grouped GEMM.

Small-T routing remains dynamic data inside fixed launch shapes. It must cover:

- autoregressive proposal steps;
- target candidate-window verification;
- eager execution;
- captured MTP round execution.

MTP prompt preparation is not a small-T path. It follows the ordinary prompt in bounded
`mtp_prefill_chunk` slices and must use the grouped/prefill MoE contract in the next section.

### 8.5 Prefill

For prefill, each token creates `top_k` token-expert assignments. The operator requires bounded
workspace for at least:

```text
topk_ids[T,top_k]
topk_weights[T,top_k]
expert_counts[E]
expert_offsets[E+1]
token/expert permutation[T*top_k]
reverse or reduction mapping[T*top_k]
expert intermediate storage or streamed tiles
routed output accumulation[T,H]
shared expert intermediates
```

The implementation groups assignments by expert and invokes grouped or persistent expert GEMM
without unconditional host launches for all 256 experts. It must handle:

- uniform expert distribution;
- all tokens concentrated into the same eight experts;
- empty experts;
- repeated execution across chunk boundaries;
- `T*top_k` and workspace-size overflow checks.

No expert capacity limit may silently drop assignments.

### 8.6 Existing operator expansion

In addition to MoE, current operators and dispatch need explicit 35B coverage for:

- hidden width 2048 norms and residual operations;
- GQA 16Q/2KV with head dimension 256;
- GDN 16 QK heads and 32 V heads, including V:QK mapping ratio two;
- GDN Q/K/V projection rows 8192 plus Z rows 4096 (12288 fused-dispatch total), convolution Q/K/V
  rows 8192, and separate `[32,2048]` A/B controls;
- low-bit linears with new `(N,K)` shapes;
- `lm_head[248320,2048]` and draft heads with `K=2048`;
- MTP attention and fc shapes;
- Vision merger `[2048,4608]` and visual scatter width 2048.

Existing tuned 27B shape branches stay in the finite dispatch plan. Correct 35B fallbacks are added
first; tuning follows evidence.

## 9. State, Workspace, And CUDA Graphs

### 9.1 Persistent state

Profile-owned state geometry includes:

- exact full-attention layer count and slot map;
- KV heads and head dimension;
- exact GDN layer count and slot map;
- GDN convolution, QK, V, and recurrent dimensions;
- MTP KV geometry;
- MTP and turn-boundary GDN snapshot-slot count;
- vocab/hidden-sized stable step and sampling buffers;
- stable MoE routing ids, weights, and counters where graph capture requires them.

Prefix reuse and MTP commit semantics remain common. They use the profile's slot maps rather than
assuming 16 KV layers and 48 GDN layers.

### 9.2 Workspace planning

Dense and MoE workspace are different model contracts. The MoE plan covers decode, small-T,
prefill, and MTP peaks independently and takes the maximum only after exact allocation sequences are
known.

The default remains a function of prefill chunk, profile, and enabled features, not total context.
Workspace high-water reports must remain flat when prompt length grows at fixed prefill chunk.

### 9.3 CUDA Graph stability

Dynamic expert ids are graph data, not graph structure. Graph safety requires:

- stable expert-bank payload addresses;
- stable router/top-k/output buffers;
- fixed launch counts and maximum shapes per captured path;
- no host-side expert branching;
- no post-capture movement of EngineImpl, cards, arenas, or sampling state;
- graph reset before unloading or changing model profile.

Ordinary decode and full MTP rounds retain separate graph objects. Eager and captured execution must
share the same mathematical schedule.

### 9.4 Memory qualification

The provisional 20.02 GiB artifact estimate leaves plausible space on a 32 GiB device because 35B
KV/GDN state is smaller than 27B, but this is not yet a supported memory point.

Qualification records, for both BF16 and INT8 KV where applicable:

- CUDA runtime and non-arena overhead;
- selected q5090 module residency;
- context-dependent KV bytes;
- GDN snapshot slots;
- Text and Vision workspace capacity and peak;
- graph instantiation memory;
- tokenizer/media host memory where relevant;
- success or rejection of complete prefill/decode/MTP flows.

## 10. Python Reference And Diagnostics

The Python reference currently has one global 27B card in
[`../../tools/q5090/ref/config.py`](../../tools/q5090/ref/config.py), and its Text/MTP schedules always
run dense FFNs. It must gain the same two exact profile identities while remaining an independent
mathematical implementation.

Required changes include:

- v5 ModelRecord and expert-bank parsing;
- dense and MoE reference FFNs;
- explicit FP32 router softmax and top-k normalization;
- profile-specific Text, MTP, state, and Vision geometry;
- expert-bank dequantization by selected expert;
- taps for router logits, selected ids/weights, routed output, shared output, and layer output;
- real-weight block and layer parity for both profiles;
- profile identity in diagnostic dumps.

The CUDA algorithm must not be copied into the Python oracle. The reference follows upstream model
mathematics and documented BF16 rounding boundaries.

## 11. Tokenizer, Templates, Serving, And Reports

### 11.1 Runtime assets

v5 continues to embed checkpoint-specific tokenizer and generation assets. It additionally records:

- chat-template profile id and source-template hash;
- image/video processor profile ids and source hashes;
- normalized special token identities;
- any raw template/processor assets retained for provenance.

The runtime uses an explicit compiled template/processor profile. It does not execute arbitrary
Jinja or arbitrary JSON behavior. The raw assets make provenance and parity auditable.

### 11.2 Chat rendering

The existing C++ renderer contains checkpoint-specific system/developer and thinking-history
behavior. Family work must compare each compiled template profile to its source Jinja on canonical
text, tools, thinking-preservation, system-role, image, video, and tool-result fixtures.

Protocol adapters may intentionally normalize external roles before rendering, for example folding
OpenAI `developer` or Anthropic system reminders. Such behavior remains a documented serving
contract and is not presented as literal source-Jinja behavior.

### 11.3 Model identity

The loaded artifact supplies the canonical model id. Serving behavior becomes:

- `/v1/models` lists exactly the loaded canonical id;
- `/v1/models/{id}` accepts exactly that canonical id;
- request `model` must equal the loaded canonical id;
- CLI load summaries print model and packing profile;
- no default `qwen3.6-27b` identity is silently used for a 35B artifact.

The initial family surface has no serving aliases. The current project-owned `--model-id` override
is removed because it can make the server advertise an identity different from the loaded artifact.

Any change to OpenAI/Anthropic schema behavior is updated in schema tests and
[`../serving.md`](../serving.md) in the same phase.

### 11.4 Benchmark and profiler identity

`qus_bench` JSON/CSV/table output adds:

- model family and exact profile;
- packing profile;
- artifact format version and source-index hash;
- resident expert-bank bytes;
- active expert bytes per token where computable;
- MoE execution regime and routing configuration.

27B and 35B reports are distinct baselines. No performance statement compares reports that differ
in model, artifact, commit, dirty state, command, context, KV dtype, graph mode, or MTP mode without
making that difference explicit.

## 12. Ownership And Source Impact

| Area | Owner | Required change |
|---|---|---|
| profile registry | L2/runtime boundary | exact specs, immutable ModelInfo, factory |
| q5090 v5 parser | L0 | generic structural safety and bounded metadata |
| profile schema validator | L2-facing artifact validation | exact model/packing inventory before CUDA |
| Weight / expert bank | L0 | non-owning direct-consumption bank descriptor |
| converter and verifier | offline tools | normalized profiles, v5 writer, bank streaming, both artifacts |
| linear codec/dispatch | L1 | expert layout addressing and new finite shape families |
| router and MoE | L1 | mathematical APIs, decode/small-T/prefill launchers |
| Text/MTP schedules | L2 | dense/MoE FFN structures and concrete cards |
| cache/state/work planner | runtime + L0 sizing helpers | profile-derived exact allocation |
| Vision | L2 | profile output width and binding validation |
| processor | media/model boundary | explicit multimodal profile |
| tokenizer/template | text | checkpoint/profile-aware assets and renderer |
| OpenAI/Anthropic | serve | loaded model identity and template-visible behavior |
| Python reference/parity | tools | v5/expert/profile support and independent oracle |
| benchmarks/profilers | bench | MoE entrypoints, profile identity, new matrices |
| active documentation | docs | family scope, both profiles, v5, serving and kernel workflow |

L1 must remove any dependency on the global model card. Operator wrappers validate relationships
among their arguments or receive explicit semantic dimensions; they do not include `model::kCfg`.

## 13. Shared Coordination Points

The following contracts must be frozen before parallel implementation can safely diverge:

1. normalized profile ids and exact fields;
2. ModelRecord and LayerRecord byte layouts;
3. MoE source-role and fusion ids;
4. final expert-bank physical layout;
5. router and expert numerical rounding contract;
6. `ExpertWeightBank` addressing semantics;
7. public MoE operator inputs/outputs and workspace ownership;
8. 35B packing/qtype policy;
9. profile/template/processor identity rules;
10. benchmark/report identity fields.

Converter, parser, Python reader, C++ loader, model binding, kernels, and tests must consume the same
frozen definitions. Temporary duplicate definitions used during development must be reconciled before
the phase is complete.

## 14. Dependencies And Delivery Sequence

The dependency chain is:

```text
exact model facts
  -> normalized profiles and candidate v5 ABI
  -> bank-layout microbenchmark and final layout
  -> v5 ABI freeze
  -> converter/parser/reader descriptors
  -> real 35B artifact
  -> MoE numerical operators
  -> 35B Text schedule and resource plan
  -> MTP/Vision/frontends
  -> graph/performance qualification
  -> active-doc replacement and plan archive
```

Kernel prototypes may use synthetic expert banks before the real artifact exists, but no final
kernel or converter contract may invent a different bank layout independently.

## 15. Implementation Phases

### Phase 0: Freeze the source facts and 27B baseline

**Behavioral boundary:** later migration and performance claims have an immutable source identity,
an approved numerical oracle, and a reproducible pre-change 27B baseline.

Tasks:

- record the config and safetensors-index hashes for both local BF16 checkpoints;
- record the current v4.2 artifact hash, manifest, converter identity, git commit, dirty state, GPU,
  CUDA, and complete commands;
- add a small diagnostic activation-tap harness around the existing model `FileTap` and independent
  HF reference, then capture current 27B mixer/FFN/layer/final-logit evidence;
- capture the current 27B weight/output parity evidence and the `smoke` and `core` benchmark matrices;
- freeze the 27B numerical tolerances and the allowed performance/memory regression thresholds
  before introducing the family boundary;
- state which long-running real-artifact gates are mandatory at each later milestone.

Verification and evidence capture:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL27=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
MODEL35=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS27_V4=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
BASELINE=profiles/bench/qwen36-family-baseline

test -d "$MODEL27" && test -d "$MODEL35" && test -s "$WEIGHTS27_V4"
mkdir -p "$BASELINE"
git rev-parse HEAD > "$BASELINE/git-commit.txt"
git status --short > "$BASELINE/git-status.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
  > "$BASELINE/gpu.txt"
nvcc --version > "$BASELINE/cuda-toolkit.txt"
nsys --version > "$BASELINE/nsys-version.txt"
ncu --version > "$BASELINE/ncu-version.txt"
sha256sum "$MODEL27/config.json" "$MODEL27/model.safetensors.index.json" \
  "$MODEL35/config.json" "$MODEL35/model.safetensors.index.json" \
  "$WEIGHTS27_V4" "$WEIGHTS27_V4.manifest.json" > "$BASELINE/input-sha256.txt"

$PYTHON -m tools.q5090_convert.verify "$WEIGHTS27_V4" --model "$MODEL27"
$PYTHON tools/parity/block_parity.py --weights "$WEIGHTS27_V4" --hf "$MODEL27"
cmake --build build -j --target qus_profile_tap
$PYTHON tools/parity/run_profile_parity.py --weights "$WEIGHTS27_V4" --hf "$MODEL27" \
  --cpp ./build/tests/qus_profile_tap --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir "$BASELINE/activation-parity"
./build/src/qus "$WEIGHTS27_V4" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids \
  > "$BASELINE/greedy.stdout" 2> "$BASELINE/greedy.stderr"
$PYTHON tools/bench/run_qus_bench_matrix.py --preset smoke --weights "$WEIGHTS27_V4" \
  --output-dir "$BASELINE/smoke"
$PYTHON tools/bench/run_qus_bench_matrix.py --preset core --weights "$WEIGHTS27_V4" \
  --output-dir "$BASELINE/core"
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none \
  -o "$BASELINE/qwen36-27b-v4-k3" ./build/bench/qus_bench --weights "$WEIGHTS27_V4" \
  -pg 2048,128 -r 1 --warmup 1 --mtp-draft-tokens 3
```

The `core` matrix and real-model parity are deliberate milestone work, not ordinary edit-loop tests.

Definition of done:

- every later before/after claim names this exact artifact and source identity;
- numerical, memory, latency, and throughput acceptance thresholds are recorded before migration;
- no archived profiler report is treated as the current baseline.

### Phase 1: Freeze profiles and q5090 v5

**Behavioral boundary:** one exact artifact identity selects one exact compiled profile, and the v5
binary contract can describe dense and MoE topology without executing either model.

Tasks:

- define the two normalized model profiles and exact layer tables;
- write a self-contained candidate v5 binary specification owned by this plan, without calling it
  current or normative before the Phase 2 cutover;
- define ModelRecord, LayerRecord, profile/packing ids, source roles, and candidate expert layout;
- define the structural/profile validation split and its malformed-input matrix without adding a
  second accepted parser to the current runtime;
- define malformed-input and profile-mismatch requirements;
- define v5 manifest/report identity;
- implement a self-contained expert-layout benchmark and make an evidence-backed final layout choice
  at the real gate-up/down bank shapes.

Verification:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target qus_expert_layout_bench
./build/bench/qus_expert_layout_bench --layouts bank-local,flat-global \
  --experts 256 --rows 1024 --cols 2048 --qtype q4g64 \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
./build/bench/qus_expert_layout_bench --layouts bank-local,flat-global \
  --experts 256 --rows 2048 --cols 512 --qtype q5g64 \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
git diff --check
```

Definition of done:

- the candidate v5 contract is self-contained and implementable;
- the registry recognizes only the two intended profiles;
- parser safety does not depend on trusting a profile;
- profile validity does not depend on tensor names;
- the bank layout has microbenchmark evidence at real shapes.

### Phase 2: Migrate the tooling and 27B artifact

**Behavioral boundary:** the existing 27B model runs exclusively from a v5 artifact through the new
profile boundary with unchanged model behavior.

Tasks:

- implement v5 serializer, reader, verifier, and manifest;
- convert tensor-plan globals into profile inputs;
- convert parser/loader/WeightStore to v5 only;
- implement structural and exact-profile validators together with the frozen malformed-input matrix;
- implement the Engine profile factory with only the dense profile enabled initially;
- move cache/work/step planning behind the dense profile;
- regenerate the 27B v5 artifact;
- promote the candidate v5 specification and archive v4 in the same finishing change that makes v5
  the only accepted format; update every active link at that cutover;
- remove v4 runtime/converter branches and retired fixtures;
- update `AGENTS.md` and every real-test, MTP/Vision E2E, benchmark-runner, tool, and documentation
  consumer of the canonical 27B artifact path in that same cutover;
- replace the misleading weight-only `block_parity.py` name with a v5 weight-parity tool and port the
  Phase 0 C++/Python activation-tap driver to profile-aware v5 block, layer, and final-logit parity;
- make real-file tests accept an explicit artifact/profile and register a `profile-27b` CTest label;
- replace fixture tests' ambient `python3` subprocesses with the canonical Python selected by CMake.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
WEIGHTS=out/qwen3_6_27b.q5090_mixed_v5.qus

mkdir -p profiles/bench
$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
$PYTHON -m tools.q5090_convert.convert --model "$MODEL" --tokenizer "$MODEL" --out "$WEIGHTS"
test -s "$WEIGHTS" && test -s "$WEIGHTS.manifest.json"
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --quick
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --model "$MODEL"
$PYTHON tools/parity/weight_parity.py --weights "$WEIGHTS" --hf "$MODEL"

cmake --build build -j --target qus_q5090_parser_test qus_q5090_pack_golden_test \
  qus_weight_store_test qus_weight_store_real_file_test qus_model_bind_test \
  qus_model_blocks_test qus_engine_real_file_test qus_profile_tap qus_bench
ctest --test-dir build -R \
  'qus_(q5090_parser|q5090_pack_golden|weight_store|weight_store_real_file|model_bind|model_blocks|engine_real_file)_test' \
  --output-on-failure
ctest --test-dir build -L profile-27b --output-on-failure
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_dense_27b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir profiles/bench/qwen36-27b-v5-parity

cmake --build build -j
ctest --test-dir build --output-on-failure
$PYTHON tools/bench/run_qus_bench_matrix.py --preset core --weights "$WEIGHTS" \
  --output-dir profiles/bench/qwen36-27b-v5-core

rg -n --glob '!docs/archive/**' \
  '(q5090[_ -]?v4|format_v4|qwen3_6_27b\.q5090_w4g64_mixed_v4_2)' \
  AGENTS.md README.md docs bench tests tools
```

The parser/loader suites include valid dense and MoE model records; unknown profiles; wrong layer
tables; top-k zero/overflow; bank rank/axis/stride/payload overflow; missing or duplicate roles;
wrong Text/MTP/Vision layer bounds; and Vision/draft hidden mismatch. They also prove that a
structurally safe file is rejected when it fails the selected exact profile.

Definition of done:

- no executable or tool accepts v4;
- the active format specification is v5, the v4 specification is archived, and no active link calls
  v4 current;
- all canonical 27B consumers use the explicit v5 path and a full `ctest` has exercised the cutover;
- 27B block/model/end-to-end behavior matches the approved pre-migration oracle;
- 27B cache/workspace high-water behavior remains bounded by prefill chunk;
- the before/after `qus_bench` matrix shows no unexplained regression;
- report output names the exact model and packing profile.

### Phase 3: Implement expert-bank tooling and MoE operators

**Behavioral boundary:** a synthetic or converted expert bank produces numerically correct Qwen MoE
outputs for decode, small-T, and prefill without host-selected routing.

Tasks:

- implement bounded expert-bank encode/decode/verification;
- implement `ExpertWeightBank` and selected-expert views/addressing;
- add one risk-centered `qus_sparse_moe_test` contract suite with independent router, selected
  expert, shared expert, and reduction oracles;
- add shared expert and routed expert contracts;
- implement correct fallbacks for T=1, T=2..6, and prefill;
- add per-operator benchmarks for routing and expert execution;
- add real 35B attention/GDN/linear shapes to existing dispatch tests and benchmarks;
- run sanitizer coverage for first/last expert, skewed routing, and repeated workspace reuse.

Intended verification targets:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python

$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
cmake --build build -j --target qus_sparse_moe_test qus_sparse_moe_bench \
  qus_q5090_parser_test qus_q5090_pack_golden_test qus_weight_store_test qus_linear_test \
  qus_rmsnorm_test qus_gqa_attention_test qus_gdn_gating_test qus_gdn_gating_proj_test \
  qus_gated_delta_rule_test qus_causal_conv1d_silu_test
ctest --test-dir build -R \
  'qus_(sparse_moe|q5090_parser|q5090_pack_golden|weight_store|linear|rmsnorm|gqa_attention|gdn_gating|gdn_gating_proj|gated_delta_rule|causal_conv1d_silu)_test' \
  --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/tests/qus_sparse_moe_test
compute-sanitizer --tool racecheck --error-exitcode=99 ./build/tests/qus_sparse_moe_test
./build/bench/qus_sparse_moe_bench --profile qwen3_6_35b_a3b \
  --tokens 1,2,6,128,1024 --routing uniform,skewed
```

If grouped prefill introduces count/offset/permutation storage that can expose uninitialized reads,
the same suite also runs under `compute-sanitizer --tool initcheck`.

Numerical cases include:

- exact router ids/weights away from and near top-k boundaries;
- deterministic tie behavior;
- selected expert ids 0 and 255;
- all token assignments concentrated in eight experts;
- uniform and sparse expert histograms;
- independent gate/up/down and shared-gate oracle comparison;
- BF16 rounding and accumulation stress;
- repeated arena mark/rewind and graph-stable addresses.

Definition of done:

- the public contract contains no CUDA strategy fields;
- no hot path reads routing state to the host;
- every regime matches the independent oracle within its approved tolerance;
- sanitizer finds no OOB, stale workspace, or bank-stride error;
- benchmark output identifies layout, qtypes, T regime, and active bytes.

### Phase 4: Produce and bind the real 35B Text artifact

**Behavioral boundary:** 35B Text prefill and ordinary eager decode run from the real packed artifact
with correct KV/GDN state and no MTP or Vision dependency.

Tasks:

- implement the MoE Text tensor plan and apply the Phase 2 exact-profile validator to the complete
  real inventory;
- stream-convert the real 35B source;
- evaluate candidate expert/shared qtypes at weight, activation, layer, and short-generation levels;
  freeze the 35B packing profile id only after acceptance and regenerate the canonical artifact if
  the accepted policy differs from the candidate;
- implement the MoE model card and weight binding;
- implement profile-derived KV, GDN, StepState, and workspace planning;
- add 35B attention/GDN slot maps;
- implement Text prefill, greedy decode, sampling, and prefix reset/append behavior;
- establish block, layer, and short-generation parity.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

$PYTHON -m py_compile $(rg --files tools/q5090_convert tools/q5090 tools/parity | rg '\.py$')
$PYTHON -m pytest tools/q5090_convert/tests tools/q5090/tests
$PYTHON -m tools.q5090_convert.convert --model "$MODEL" --tokenizer "$MODEL" --out "$WEIGHTS"
test -s "$WEIGHTS" && test -s "$WEIGHTS.manifest.json"
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --quick
$PYTHON -m tools.q5090_convert.verify "$WEIGHTS" --model "$MODEL"
$PYTHON tools/parity/weight_parity.py --weights "$WEIGHTS" --hf "$MODEL" --rows-per-chunk 256

cmake --build build -j --target qus_q5090_parser_test qus_q5090_pack_golden_test \
  qus_weight_store_test qus_weight_store_real_file_test qus_model_bind_test qus_model_blocks_test \
  qus_engine_real_file_test qus_runtime_file_tap_test qus_profile_tap qus_bench
ctest --test-dir build -R \
  'qus_(q5090_parser|q5090_pack_golden|weight_store|weight_store_real_file|model_bind|model_blocks|engine_real_file|runtime_file_tap)_test' \
  --output-on-failure
ctest --test-dir build -L profile-35b-a3b --output-on-failure
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_moe_35b_a3b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases prefill,decode \
  --output-dir profiles/bench/qwen36-35b-a3b-text-parity
./build/bench/qus_bench --weights "$WEIGHTS" -p 128,512,2048 -n 128 \
  -pg '2048,128' -r 5 --warmup 1 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-text-eager.json --no-cuda-graph
```

Definition of done:

- the complete artifact passes structural, CRC, padding, plan, and value verification;
- every Text layer has exactly one valid router, two banks, one shared expert, and one shared gate;
- eager prefill/decode matches the approved Python/HF oracle at block, layer, and observable output
  levels;
- weight-only parity is reported separately and is never used as evidence for activation parity;
- the canonical artifact names a packing profile whose qtype policy has passed the accepted quality
  gates;
- cache/state/workspace planning matches actual high-water use;
- no 27B constant is used to interpret a 35B tensor or state slot.

### Phase 5: Add CUDA Graphs and qualify MoE performance

**Behavioral boundary:** ordinary 35B decode and Text prefill meet correctness and stable-address
requirements under their production execution paths.

Tasks:

- capture/replay ordinary MoE decode;
- optimize selected-expert decode only after NSYS attribution;
- optimize small-T and grouped prefill paths only from benchmark evidence;
- profile the frozen bank layout in complete inference and tune kernels without changing its v5
  addressing contract;
- validate BF16 and INT8 KV where supported;
- add stable NVTX ranges for routing, routed experts, shared expert, and reduction if useful to
  whole-inference attribution.

Verification:

```bash
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus_moe_graph_replay_test qus_engine_real_file_test qus qus_bench
ctest --test-dir build -R 'qus_(moe_graph_replay|engine_real_file)_test' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/tests/qus_moe_graph_replay_test

./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --no-cuda-graph \
  > /tmp/qwen36-35b-eager.out 2> /tmp/qwen36-35b-eager.err
./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids \
  > /tmp/qwen36-35b-graph.out 2> /tmp/qwen36-35b-graph.err
diff -u /tmp/qwen36-35b-eager.out /tmp/qwen36-35b-graph.out

./build/bench/qus_bench --weights "$WEIGHTS" -p 128,512,2048 -n 128 \
  -pg '2048,128' -r 5 --warmup 1 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-graph.json
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none \
  --output profiles/nsys/qwen36-35b-a3b-decode \
  ./build/bench/qus_bench --weights "$WEIGHTS" -pg 2048,128 -r 1 --warmup 1
```

`qus_moe_graph_replay_test` captures once, then replays repeated inputs that force different and
highly skewed expert sets while checking stable addresses and eager/replay numerical equivalence.
The real CLI diff proves observable greedy equivalence for the packed 35B artifact; neither check is
replaced by the latency benchmark.

NCU commands are selected only after the NSYS report identifies a specific kernel. The exact kernel
shape and metric set are recorded with the resulting report rather than predeclared generically.

Definition of done:

- eager and graph outputs agree;
- expert identity changes do not trigger graph recapture or host branching;
- no capture-invalid allocation or address movement occurs;
- NSYS explains the caller-visible Text latency breakdown;
- any kernel performance claim has matching NCU and end-to-end `qus_bench` evidence.

### Phase 6: Add MoE MTP and Vision

**Behavioral boundary:** the complete 35B Text/MTP/Vision model works under eager and graph execution,
and speculative decoding preserves the target distribution.

Tasks:

- bind and execute the MoE MTP layer in prompt, proposal, and verify modes;
- route MTP prompt preparation through the grouped/prefill MoE path at bounded
  `mtp_prefill_chunk`, while proposal and verification use the small-T path;
- allocate MTP KV and GDN snapshot slots from the profile;
- capture/replay full MTP rounds;
- regenerate and evaluate a width-2048 shortlisted draft head;
- bind the 35B Vision weights and width-2048 merger;
- add one canonical small RGB Vision fixture shared by the C++ and Python activation-parity drivers;
- validate visual scatter and MRoPE continuation;
- preserve target correctness independently of draft-head quality.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16
WEIGHTS=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus_engine_mtp_e2e_test qus_engine_vision_e2e_test \
  qus_mtp_round_test qus_mtp_prefill_moe_test qus_vision_support_test qus_profile_tap qus qus_bench
ctest --test-dir build -R \
  'qus_(engine_mtp_e2e|engine_vision_e2e|mtp_round|mtp_prefill_moe|vision_support)_test' \
  --output-on-failure
ctest --test-dir build -L profile-35b-a3b \
  -R 'engine_(mtp|vision)_e2e' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/tests/qus_engine_mtp_e2e_test
$PYTHON tools/parity/run_profile_parity.py --profile qwen3_6_moe_35b_a3b_v1 \
  --weights "$WEIGHTS" --hf "$MODEL" --cpp ./build/tests/qus_profile_tap \
  --token-ids 1,2,3,4 --phases mtp-prefill,mtp-propose,mtp-verify,vision \
  --vision-fixture tests/fixtures/media/qwen36_vision_reference.ppm \
  --output-dir profiles/bench/qwen36-35b-a3b-mtp-vision-parity

./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --mtp-draft-tokens 3 --no-cuda-graph \
  > /tmp/qwen36-35b-mtp-eager.out 2> /tmp/qwen36-35b-mtp-eager.err
./build/src/qus "$WEIGHTS" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids --mtp-draft-tokens 3 \
  > /tmp/qwen36-35b-mtp-graph.out 2> /tmp/qwen36-35b-mtp-graph.err
diff -u /tmp/qwen36-35b-mtp-eager.out /tmp/qwen36-35b-mtp-graph.out

./build/bench/qus_bench --weights "$WEIGHTS" -p 512 -n 256 -r 5 --warmup 1 \
  --mtp-draft-tokens 5 -o json \
  --output-file profiles/bench/qwen36-35b-a3b-mtp-full-head.json
./build/bench/qus_bench --weights "$WEIGHTS" -p 512 -n 256 -r 5 --warmup 1 \
  --mtp-draft-tokens 5 --lm-head-draft -o json \
  --output-file profiles/bench/qwen36-35b-a3b-mtp-draft-head.json
```

Definition of done:

- MTP target verification and state-slot commit are correct for rejection at every candidate
  position and context-capacity fallback;
- MTP prompt preparation matches the reference across multiple chunk boundaries and never uses the
  small-T proposal path;
- sampled MTP preserves the target distribution;
- the draft head affects acceptance/performance only;
- Vision block/merger output matches the independent reference;
- visual embeddings are exactly width 2048 and decode continues with correct MRoPE state;
- sanitizer finds no state-slot, rewind, or multimodal workspace lifetime error.

### Phase 7: Complete frontends, serving, reports, and active docs

**Behavioral boundary:** both profiles are complete user-visible products under CLI, OpenAI, and
Anthropic surfaces, and active documentation describes only implemented behavior.

Tasks:

- implement and verify checkpoint-specific template/processor profiles;
- report the loaded canonical model id in CLI and HTTP model endpoints;
- remove the caller-selected `--model-id` override and reject every request/lookup identity other
  than the artifact's canonical id;
- update tokenizer/template/processor fixtures where they protect observable behavior;
- update benchmark report schema and downstream tests;
- add a stdlib-only loopback serving smoke driver that starts one server at a time and verifies
  model listing/lookup, wrong-model rejection, OpenAI, Anthropic, tools, thinking, and image cases;
- update README, design, family model reference, kernel guide, serving guide, tools guides, and test
  guides;
- update `AGENTS.md` canonical paths for both supported real artifacts;
- archive this completed plan and its dated evidence.

Verification:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
WEIGHTS27=out/qwen3_6_27b.q5090_mixed_v5.qus
WEIGHTS35=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus

cmake --build build -j --target qus qus-serve qus_qwen_chat_template_test \
  qus_qwen_text_tokenizer_test qus_qwen_text_runner_test qus_processor_test \
  qus_openai_schema_test qus_anthropic_schema_test qus_bench_support_test
ctest --test-dir build -R \
  'qus_(qwen_chat_template|qwen_text_tokenizer|qwen_text_runner|processor|openai_schema|anthropic_schema|bench_support)_test' \
  --output-on-failure
ctest --test-dir build --output-on-failure

./build/src/qus "$WEIGHTS27" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids > /tmp/qwen36-27b-cli.out
./build/src/qus "$WEIGHTS35" --prompt '你好' --max-context 256 --max-new 16 \
  --no-thinking --greedy --print-token-ids > /tmp/qwen36-35b-a3b-cli.out

$PYTHON tools/serving/family_smoke.py --server ./build/src/qus-serve \
  --weights "$WEIGHTS27" --model qwen3.6-27b --port 18080 \
  --cases models,identity,openai,anthropic,tools,thinking,image \
  --image tests/fixtures/media/qwen36_vision_reference.ppm
$PYTHON tools/serving/family_smoke.py --server ./build/src/qus-serve \
  --weights "$WEIGHTS35" --model qwen3.6-35b-a3b --port 18081 \
  --cases models,identity,openai,anthropic,tools,thinking,image \
  --image tests/fixtures/media/qwen36_vision_reference.ppm

rg -n --glob '!docs/archive/**' \
  '(q5090[_ -]?v4|format_v4|fixed Qwen3\.6-27B|only.*27B|single.*27B)' \
  AGENTS.md README.md docs bench tests tools

$PYTHON - <<'PY'
import re
import subprocess
import urllib.parse
from pathlib import Path

files = [
    Path(p)
    for p in subprocess.check_output(["git", "ls-files", "*.md"], text=True).splitlines()
    if not p.startswith("docs/archive/")
]
pattern = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\s\n]+)")
missing = []
for source in files:
    text = re.sub(r"```.*?```", "", source.read_text(encoding="utf-8"), flags=re.S)
    for raw in pattern.findall(text):
        target = raw.strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local = urllib.parse.unquote(target.split("#", 1)[0])
        if local and not (source.parent / local).exists():
            missing.append((source, target))
for source, target in missing:
    print(f"{source}: missing local link {target}")
raise SystemExit(bool(missing))
PY

git diff --check
```

Before running these commands, compare them with each newly built executable's current `--help` and
update this plan if the implemented option surface deliberately changed.

Definition of done:

- each artifact uses its own tokenizer/template/processor identity;
- CLI and server advertise the actual loaded model;
- OpenAI/Anthropic behavior and documentation agree;
- report consumers can distinguish model and packing profiles;
- all active documentation links resolve and no active document calls v4 or fixed-27B behavior
  current;
- the plan and evidence are archived.

## 16. Test And Evidence Matrix

Only tests that protect an allowed project risk are added.

| Risk | Required evidence |
|---|---|
| v5 bounds/overflow/layout | binary parser malformed-input tests |
| expert bank code/scale addressing | bit-identical quantizer/decoder oracle at real shapes |
| profile mismatch | parser/schema rejection fixtures |
| router/top-k | FP32 mathematical oracle, boundary/tie stress |
| routed/shared MoE | BF16-rounded operator oracle for T=1, small-T, prefill |
| attention/GDN new geometry | existing operator oracles at 35B real shapes |
| layer schedule | block/layer parity and observable model output, not source scans |
| workspace/state lifetime | repeated execution, high-water reports, compute-sanitizer |
| CUDA Graph | eager/replay output equivalence and stable-address execution |
| MTP state/target correctness | rejection-position, capacity, sampled-distribution E2E cases |
| Vision width/MRoPE | real Vision/processor parity and multimodal E2E behavior |
| tokenizer/template | canonical rendered-token fixtures per profile |
| serving | OpenAI/Anthropic schema tests and real responses |
| performance | per-op bench, NSYS, targeted NCU, before/after `qus_bench` |

Tests must not scan source for preferred symbols, lock private class layout, preserve v4 compatibility,
or duplicate coverage merely because a second profile exists. Shared tests should be table-driven only
where the observable contract is genuinely shared; profile-specific numerical fixtures remain
explicit. In particular, family work must not grow a model-config test into a list of trivial
constant assertions; exact dimensions are protected by artifact/profile schema rejection, real
weight binding, and numerical model execution. Any old constant-only coverage that protects no
observable risk is removed rather than replicated for 35B.

### 16.1 Real-artifact gate policy

The current real-file tests are not yet a family gate: they hard-code the 27B v4.2 path and return
success after printing `SKIP` when an artifact, GPU, or sufficient memory is absent. Phase 2 must
replace that behavior with an explicit registration contract:

- every real-file executable accepts `--weights <path>` and `--expect-profile <id>`;
- CMake registers separately named tests with `profile-27b` or `profile-35b-a3b` labels and explicit
  canonical artifact paths; no test chooses a file by glob, modification time, or `latest`;
- a developer without a large local prerequisite may still receive a clearly reported skip;
- a formal milestone/release gate first proves both artifacts exist and then treats every skip as a
  failure;
- fixture generators receive the canonical Python 3.11 executable from CMake instead of invoking an
  ambient `python3`.

A formal two-profile gate therefore includes:

```bash
WEIGHTS27=out/qwen3_6_27b.q5090_mixed_v5.qus
WEIGHTS35=out/qwen3_6_35b_a3b.q5090_mixed_v5.qus
LOGDIR=profiles/bench/qwen36-family-final

test -s "$WEIGHTS27" && test -s "$WEIGHTS27.manifest.json"
test -s "$WEIGHTS35" && test -s "$WEIGHTS35.manifest.json"
mkdir -p "$LOGDIR"
set -o pipefail
ctest --test-dir build -V -L 'profile-27b|profile-35b-a3b' | tee "$LOGDIR/ctest.log"

if rg -n 'SKIP:' "$LOGDIR/ctest.log"; then
  echo 'formal family gate contained a skipped test' >&2
  exit 1
fi
```

Ordinary `ctest` success is not evidence that either real model was loaded unless its verbose log
shows the requested profile and artifact and contains no skip.

## 17. Major Risks And Mitigations

### 17.1 Parser generalization accepts the wrong model

**Risk:** removing fixed 27B checks turns a shape mismatch into silent wrong-weight execution.

**Mitigation:** retain strict L0 structure checks and add a separate exact profile/inventory
validator before any CUDA allocation or upload.

### 17.2 Family abstraction regresses 27B

**Risk:** runtime dimensions, virtual layer calls, or generic kernels increase launch gaps or reduce
specialization.

**Mitigation:** dispatch once around a whole concrete EngineImpl; retain 27B finite shape branches;
require a before/after 27B NSYS/`qus_bench` gate.

### 17.3 Expert layout is frozen before evidence

**Risk:** a convenient binary layout creates scattered selected-expert loads or poor prefill GEMM.

**Mitigation:** benchmark bank-local and flat-global candidates at the two real expert shapes before
v5 is normative.

### 17.4 Router quantization changes discrete expert identity

**Risk:** small linear error produces a different top-k set and large downstream divergence.

**Mitigation:** keep router and shared scalar gate BF16 initially; test top-k overlap and downstream
parity before considering any later quantization.

### 17.5 MTP is accidentally treated as dense

**Risk:** target Text works while proposal preparation uses the wrong MTP FFN.

**Mitigation:** MTP owns an independent profile and exact inventory; all target/MTP modes have taps
and parity evidence.

### 17.6 Dynamic routing breaks CUDA Graphs

**Risk:** host branching, allocation, or variable launch topology invalidates replay.

**Mitigation:** ids/counts are stable-address device data; launch maximum fixed shapes; compare eager
and replay under changing expert selections.

### 17.7 Prefill workspace is underestimated

**Risk:** `T*top_k` permutation/intermediate storage overflows or makes workspace depend on total
prompt length.

**Mitigation:** profile-owned checked sizing from prefill chunk; skew/uniform routing tests; arena
high-water and sanitizer evidence.

### 17.8 GDN state mapping is wrong

**Risk:** old 48-head or ratio-three assumptions corrupt 32-head ratio-two recurrence and snapshot
slots.

**Mitigation:** add real 35B GDN oracle shapes, exact slot maps, repeated prefill/decode, and
state-snapshot parity.

### 17.9 Vision silently emits width 5120

**Risk:** merger/scatter memory corruption or wrong multimodal embeddings.

**Mitigation:** profile schema asserts `vision.out_hidden == text.hidden`; bind and E2E tests cover
the real `[2048,4608]` merger.

### 17.10 Template or tokenizer is assumed global

**Risk:** the model receives a valid token sequence with the wrong conversation semantics.

**Mitigation:** artifact-specific assets, explicit template profile/hash, and per-profile rendered-id
fixtures.

### 17.11 32 GiB feasibility is overstated

**Risk:** packed weights fit in isolation while complete context/workspace/graph residency does not.

**Mitigation:** publish only measured full-process memory matrices and reject unsupported settings
before inference.

### 17.12 Performance optimization changes numerical order

**Risk:** fused routing or reduction changes top-k weights, accumulation, or MTP acceptance.

**Mitigation:** freeze the mathematical rounding contract first; rerun operator, layer, E2E, and
target-distribution gates after every relevant optimization.

## 18. Definition Of Done

The family elevation is complete only when all of the following are true:

- the public project scope names both exact supported profiles and still excludes arbitrary models;
- only q5090 v5 is accepted and its active specification is self-contained;
- both real v5 artifacts pass structural, integrity, value, and exact inventory verification;
- the runtime selects a concrete profile from artifact identity before CUDA initialization;
- 27B retains approved numerical behavior and has no unexplained performance regression;
- 35B Text eager and graph paths pass block, layer, and observable-output parity;
- 35B MoE decode, small-T, and prefill paths have real-shape numerical and memory-safety evidence;
- 35B MTP preserves target correctness under eager/graph, greedy/sampled, and draft/full head modes;
- 35B Vision and multimodal continuation pass reference and E2E checks;
- cache/state/workspace plans match actual allocation peaks for both profiles;
- CLI, OpenAI, Anthropic, tokenizer, template, processor, and model identity behavior are coherent;
- benchmark/profiler reports name exact profile, packing, artifact, command, commit, dirty state,
  GPU, and CUDA;
- NSYS exists for complete 35B inference and NCU exists for every kernel-specific performance
  claim;
- a separate ABI, numerical, lifetime, protocol, and performance review finds no unresolved issue;
- active docs describe implemented family behavior and all completed plans/evidence are archived.

## 19. Final Review

This is a high-risk cross-cutting change and requires an independent final review after implementation.

The review is performed in five passes:

1. **ABI/security:** checked arithmetic, layout formulas, role inventory, parser bounds, tokenizer
   regions, CRC policy, and absence of v4 fallback.
2. **Numerical/model:** norm offsets, q/gate de-interleave, GDN head mapping, router/top-k, expert and
   shared FFNs, MTP target distribution, Vision merger, and BF16 rounding.
3. **Lifetime/state:** arena marks, expert descriptors, KV/GDN slots, prefix rewind, graph addresses,
   multimodal workspace, and repeated request recovery.
4. **Protocol/product:** model identity, tokenizer/template behavior, CLI, OpenAI, Anthropic, tool
   history, thinking channels, and report schemas.
5. **Performance:** 27B regression, 35B phase attribution, expert-kernel evidence, MTP acceptance,
   memory matrices, and caller-visible throughput.

Findings are resolved before the final full gate. A near-budget or difficult result is not a reason
to waive a gate.

## 20. Documentation Lifecycle And Archive Destination

While implementation is active, this file remains under `docs/plans/` and is linked from working
changes only where useful. It does not redefine current behavior.

On completion:

- stable system ownership and supported scope are integrated into `docs/design.md`;
- common family mathematics and both exact profiles replace the fixed-only model reference in one
  active authoritative document rather than parallel `v2`/`final` documents;
- q5090 v5 remains the active contract established atomically with the Phase 2 runtime cutover;
- CLI/server behavior is integrated into `docs/serving.md`;
- kernel workflow changes are integrated into `docs/kernel-development.md`;
- README, tools, tests, and bench guides are updated;
- the retired final v4.2 specification remains archived at
  `docs/archive/optimization-era/q5090_packed_file_format_v4_2.md` without overwriting the existing
  historical v4 snapshot;
- this plan and dated implementation/performance evidence move to
  `docs/archive/family-era/plans/`;
- `docs/archive/family-era/README.md` is created to describe the family expansion era, and
  `docs/archive/README.md` is updated in the same finishing change.

Abandoned or superseded work follows the same archive rule with an explicit historical status.

## 21. Focused Reading List

Before implementing a phase, read only the relevant current authorities plus this plan:

- [`../../AGENTS.md`](../../AGENTS.md) — project scope, compatibility, testing, CUDA, documentation,
  and verification policy;
- [`../design.md`](../design.md) — current ownership and runtime flows;
- [`../qwen3.6-27b-architecture.md`](../qwen3.6-27b-architecture.md) — implemented 27B mathematics,
  MTP, Vision, and state semantics;
- the local 35B `config.json`, safetensors index, and upstream Qwen3.5-MoE mathematical reference;
- [`../q5090_packed_file_format_v4.md`](../q5090_packed_file_format_v4.md) — current container,
  code-plane, segment, fusion, tokenizer, and validation contract to be replaced;
- [`../kernel-development.md`](../kernel-development.md) — L1 contract, oracle, benchmark, NSYS, and
  NCU rules;
- [`../serving.md`](../serving.md) — current CLI/OpenAI/Anthropic/template behavior;
- [`../../tools/q5090_convert/README.md`](../../tools/q5090_convert/README.md) — converter and verifier
  workflow;
- [`../../tools/q5090/README.md`](../../tools/q5090/README.md) — reference and diagnostics;
- [`../../tests/README.md`](../../tests/README.md) and [`../../bench/README.md`](../../bench/README.md)
  — allowed verification and report behavior.

Archived q5090 plans and optimization reports may explain earlier choices but are not current
requirements. Every stable requirement needed by the family implementation is restated here or in
the active authority that replaces it.
