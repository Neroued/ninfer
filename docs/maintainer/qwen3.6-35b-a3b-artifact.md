# Qwen3.6-35B-A3B Artifact and Conversion Reference

This reference records the complete `.ninfer` object inventory and conversion recipe for the exact
`qwen3.6-35b-a3b` checkpoint: object names, shapes, formats, layouts, MoE expert ordering, fused
row order, frontend resources, source transforms, memory envelope, and binder obligations. Common
framing is defined in [`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics in
[`qwen3.6-35b-a3b-model.md`](qwen3.6-35b-a3b-model.md).

## 1. Artifact identity and boundary

The registered exact model identity is:

```text
qwen3.6-35b-a3b
```

The source/tool/compiled target key is:

```text
qwen3_6_35b_a3b
```

The target key selects one exact checkpoint package. It is not serialized as another
`model_id`. The closed registry maps this `model_id` to that package before loading.

Every conforming artifact is one complete product image. It contains Text, the optimized draft
head, MTP, Vision, and the six frontend resources in this document. There are no Text-only,
no-Vision, no-MTP, alternate-quantization, GGUF-compatible, or runtime-repacked variants. The
quantization assignment is part of the exact target contract rather than a converter option.

The responsibility split is:

```text
.ninfer JSON
    model_id, object name/kind, logical stored shape, format/layout or encoding,
    payload-relative offset, and byte length

this target contract
    complete object set, exact storage signatures, expert and fused row order,
    logical views, aliases, source transforms, and binder/runtime obligations

common numeric/layout contracts
    code and scale semantics, encoder arithmetic, plane packing, padding,
    row addressing, alignment, and encoded-size calculation

exact target implementation
    immutable bindings, exact configuration, closed Variant leaves and graph frontier ranges

Qwen3.6 family runtime
    fixed layer schedule, Program state policy, workspace composition, CUDA Graph mechanics,
    and the public Engine-facing Program contract

repository-internal Ops
    semantically closed mathematical/state-transition contracts and CUDA implementations
```

The artifact JSON does not gain fields for expert count, top-k, fused roles, source names, GGUF
types, layer classes, memory budgets, GPU identity, or kernel choices. Those facts are compiled from
this contract.

## 2. Fixed checkpoint facts used by storage

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. Quantization groups
along `K`.

| Fact | Value |
|---|---:|
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| Text hidden width | 2048 |
| Text layers | 40 |
| full-attention layers | 10 |
| GDN layers | 30 |
| query heads / KV heads / head width | 16 / 2 / 256 |
| query / output-gate / K / V widths | 4096 / 4096 / 512 / 512 |
| GDN Q/K heads × width | 16 × 128 = 2048 |
| GDN V heads × width | 32 × 128 = 4096 |
| GDN Q/K/V / Z widths | 8192 / 4096 |
| GDN convolution channels / retained taps | 8192 / 3 |
| routed experts / selected experts | 256 / 8 |
| routed and shared intermediate width | 512 |
| MTP layers | 1 full-attention sparse-MoE layer |
| draft-head rows | 131072 |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 |
| Vision heads / patch input | 16 / 1536 |
| Vision merger input / output | 4608 / 2048 |
| native context capacity | 262144 |

The semantic token-id vocabulary is exactly compatible with the current 27B checkpoint. Direct
comparison establishes the same 248044-entry base token-to-id map, the same 247587 BPE merges after
normalizing their JSON representation, and the same 33 added-token definitions at ids
`248044..248076`. Both native tokenizers therefore expose the same 248077-entry `id -> token`
domain, and the same shortlist id map can be interpreted by either model. The selected local 27B
tree now uses the pinned official Qwen resources rather than the former Unsloth-expanded
serialization.

The two checkpoints use byte-identical copies of all six frontend resources in Section 4.2. They
therefore share one Qwen3.6 frontend contract. The base `tokenizer.json` carries added-token ids only
through `248069`; the shared `tokenizer_config.json` supplies ids `248070..248076`. The family
frontend must therefore merge both resources. The two models still have different weights: their
untied output heads have shapes `[248320,5120]` and `[248320,2048]`. Section 12.5 gathers the shared
shortlist rows from the 35B head.

Full-attention layers are exactly:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39
```

Every other layer is GDN. Every Text and MTP decoder layer has 256 routed experts, selects eight,
and additionally runs one gated shared expert.

Header-only inspection of the selected 26-shard checkpoint establishes 1045 BF16 source tensors
and 35,951,822,704 parameters. Main Text routed banks account for 32,212,254,720 parameters, so
expert residency and selected-expert traffic are separate first-order design constraints.

## 3. Quantization and physical-row decision

### 3.1 Local GGUF evidence and its boundary

The role-level precision decision used `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` as external evidence.

That 22,134,528,992-byte file contains 733 Text-only tensors. It has no MTP or Vision weights. Its
relevant assignments are:

| Role | GGUF assignment |
|---|---|
| token embedding | `Q8_0` |
| output head | `Q6_K` |
| large full-attention/GDN projections and shared-expert matrices | `Q8_0` |
| routed expert gate and up, all 40 layers | `Q4_K` |
| routed expert down, layers other than 34/38/39 | `Q5_K` |
| routed expert down, layers 34/38/39 | `Q6_K` |
| router, norms, and small GDN values | `F32` |

The file records an Unsloth importance-matrix conversion. Its metadata names
`Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf`, but that importance-matrix file is not present in the
local model tree, so the exact GGUF conversion is not locally reproducible from its original
inputs. It is not a conversion prerequisite for NInfer; the emitted file is descriptive evidence
for the role and layer precision tiers.

The project treats this local GGUF as a project-accepted, already exercised upstream implementation
precedent for this exact checkpoint. It does not constitute a new NInfer numeric-format admission
or a standalone public quality result. Its metadata contains no evaluation score or quality-report
URL; its precedent status comes from the project's exercised local reference, not from those
metadata fields.

GGUF `Q4_K`, `Q5_K`, and `Q6_K` are not aliases for NInfer `Q4G64_F16S`, `Q5G64_F16S`, or
`Q6G64_F16S`. Their block metadata and reconstruction rules differ. NInfer adopts the demonstrated
precision split, not GGUF bytes. The NInfer artifact uses the existing symmetric G64 schemes and
`MAXABS_F16_RECIP_RNE_V1`; implementation acceptance therefore still requires the expert-specific
numerical and behavioral evidence in Section 17.

### 3.2 Exact NInfer assignment

| Role | NInfer format | Reason |
|---|---|---|
| token embedding | `W8G32_F16S` | matches the high-precision GGUF tier and is shared by target/MTP |
| full output head | `Q6G64_F16S` | high-sensitivity full-vocabulary boundary |
| main full-attention and GDN large matrices | `W8G32_F16S` | every token executes them; preserve active-path quality |
| main shared-expert gate/up/down | `W8G32_F16S` | always active in addition to top-8 routed experts |
| routed gate/up, all 40 Text layers | `Q4G64_F16S` | dominant residency, demonstrated low-bit role tier |
| routed down, Text layers except 34/38/39 | `Q5G64_F16S` | down projection receives one extra bit |
| routed down, Text layers 34/38/39 | `Q6G64_F16S` | preserve the local importance-ranked upgrades |
| router and shared-expert scalar gate | `BF16` | preserve exact source values; routing arithmetic is qualified separately |
| Text norms, convolution, A/B projections | `BF16` | preserve exact source values |
| GDN `A_log` and `dt_bias` | `FP32` | preserve the exact expanded source values used by decay math |
| optimized draft head | `Q4G64_F16S` | proposal-only shortlist |
| MTP large matrices, including routed experts | `W8G32_F16S` | conservative within-budget tier; GGUF has no MTP evidence |
| Vision block input/expansion matrices | `Q4G64_F16S` | reuse the 27B storage/kernel geometry subject to 35B qualification |
| Vision block output/contraction matrices | `Q5G64_F16S` | reuse the 27B storage/kernel geometry subject to 35B qualification |
| Vision patch projection | `Q6G64_F16S` | reuse the 27B storage/kernel geometry subject to 35B qualification |
| Vision merger matrices | `W8G32_F16S` | preserve multimodal-to-Text boundary quality subject to 35B qualification |
| all other Vision weights/biases | `BF16` | preserve source words |

No runtime weight is stored twice for decode and prefill. The persistent bytes are directly
consumable by both regimes.

Migration from 27B means reusing compatible format, layout, and kernel geometry, not inheriting its
quality evidence. The 27B MTP block is dense while this checkpoint's MTP block is sparse MoE, and
the two checkpoints' Vision tensors are not value-identical. Section 17 therefore requires
35B-specific MTP and Vision qualification before implementation acceptance.

### 3.3 Expert-major rank-two storage

No new rank-three layout is needed. The recipe defines a semantics-preserving expert-major reshape
into the already registered rank-two `row-split-k128-v1` layout.

For source routed gate/up bank `G[E,2I,K]` with `E=256` and `I=512`, the stored matrix has shape
`[E*I*2,K] = [262144,2048]`. Each expert is half-split, matching both the source order and the
implemented 27B decode/prefill fused-SwiGLU row contract:

```text
stored_row(e, p, r) = e * 1024 + p * 512 + r
p = 0: gate row G[e,r,:]
p = 1: up   row G[e,512+r,:]
```

Expert `e` therefore owns consecutive stored rows `[e*1024,(e+1)*1024)`. A fused SwiGLU kernel
reads row `e*1024+r` and matching row `e*1024+512+r` as one gate/up pair. The existing T=1 kernel
uses this paired-row pattern, while the folded prefill MMA tile reads 32 consecutive gate rows and
the matching 32 consecutive up rows. Interleaving rows as `2r+p` would not reduce weight bytes or
HBM transactions and would prevent direct reuse of those paths.

For source routed down bank `D[E,H,I]`, the stored shape is
`[E*H,I] = [524288,512]`:

```text
stored_row(e, h) = e * 2048 + h
```

Expert `e` owns consecutive stored rows `[e*2048,(e+1)*2048)`. Logical expert identity remains the
router row id. The converter performs no expert permutation.

`row-split-k128-v1` keeps base, optional high-bit, and scale data in separate planes. An expert view
is therefore those plane spans for its consecutive rows, not one assumed-contiguous byte range.
The binder derives those spans once. Top-8 decode must address only the eight selected experts;
touching all 256 banks, gathering into another persistent weight buffer, or repacking at load time
is nonconforming.

The same half-split rule applies to the shared expert's `[1024,2048]` gate/up object and to MTP
routed/shared gate/up objects. A shared object uses `stored_row(p,r) = p*512+r`.

### 3.4 Routed-bank plane geometry

The existing global-plane `row-split-k128-v1` layout remains the persistent layout. For the exact
parent shapes in Sections 5 and 6, the following offsets and strides are fixed consequences of the
registered layout. All values are bytes relative to the start of the tensor payload; `-` means that
the format has no high-bit plane.

| Routed parent | Row bytes base/code / high / scale | Plane offsets base / high / scale | Expert strides base / high / scale |
|---|---:|---:|---:|
| main Q4 gate/up `[262144,2048]` | `0x400 / - / 0x40` | `0 / - / 0x10000000` | `0x100000 / - / 0x10000` |
| main Q5 down `[524288,512]` | `0x100 / 0x40 / 0x10` | `0 / 0x08000000 / 0x0a000000` | `0x80000 / 0x20000 / 0x8000` |
| main Q6 down `[524288,512]` | `0x100 / 0x80 / 0x10` | `0 / 0x08000000 / 0x0c000000` | `0x80000 / 0x40000 / 0x8000` |
| MTP W8 gate/up `[262144,2048]` | `0x800 / - / 0x80` | `0 / - / 0x20000000` | `0x200000 / - / 0x20000` |
| MTP W8 down `[524288,512]` | `0x200 / - / 0x20` | `0 / - / 0x10000000` | `0x100000 / - / 0x10000` |

For example, main routed gate/up uses:

```text
row       = e*1024 + p*512 + r
base_ptr  = object_base + row*0x400
scale_ptr = object_base + 0x10000000 + row*0x40
```

Main routed down uses `row = e*2048+h` and the applicable Q5/Q6 table row. These formulas permit
O(1) device addressing of every selected expert without pointer tables, gathers, or scanning the
other 248 experts. Each expert is already consecutive inside every plane. Wrapping each expert in
an independent row-split payload would not change useful bytes, transactions, or 64-KiB page
count; a new expert-local persistent layout is therefore not part of this contract. It may be
proposed later only from an identified TLB/long-scoreboard hotspot and must then beat this layout in
both NCU and end-to-end `ninfer_bench` evidence.

The corresponding useful routed-weight traffic per main layer is exactly 8.5 MiB for Q4 gate/up
plus 5.25 MiB for Q5 down or 6.25 MiB for Q6 down. Those are the decode traffic targets before
cache effects, not merely residency fractions.

### 3.5 Other fused physical rows

The converter also fixes these decode-oriented row transforms:

- full-attention input is one `[query,key,output_gate,value]` object;
- GDN input is one `[query,key,value,z]` object;
- GDN A/B projection rows are half-split `[a_0,...,a_31,b_0,...,b_31]`;
- MoE router rows `0..255` and the shared-expert sigmoid-gate row 256 form one BF16 object;
- gate/up rows are half-split as defined above.

These are physical stored objects and row orders, not serialized fusion metadata. They preserve
logical row identity and do not create a private kernel precision or rounding boundary.

## 4. Object namespace, order, and frontend resources

### 4.1 Namespace and writer order

- `text/` owns the embedding, 40 Text layers, final norm, full head, and optimized draft head.
- `mtp/` owns only MTP-specific tensors; embedding and heads alias `text/` objects.
- `vision/` owns the complete Vision tower and merger.
- `frontend/` owns the six required raw resources.

The converter writes, in order:

1. the six frontend resources below;
2. `text/token_embedding`;
3. Text layers `0..39`, using the applicable table order in Sections 5.2 and 5.3;
4. `text/final_norm`, `text/output_head`, `text/draft_head`, and
   `text/draft_head_token_ids`;
5. the fifteen MTP tensors in Section 6;
6. the Vision stem, blocks `0..26`, and merger in Section 7.

Readers bind by name, not physical array index. Names contain no format spelling.

### 4.2 Frontend resources

The artifact contains exactly these `raw-bytes-v1` resources:

| Order | Object name | Source filename | Runtime meaning |
|---:|---|---|---|
| 0 | `frontend/tokenizer.json` | `tokenizer.json` | base BPE vocabulary, merges, token bytes, and its added-token subset |
| 1 | `frontend/tokenizer_config.json` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `frontend/chat_template.jinja` | `chat_template.jinja` | registered Qwen template |
| 3 | `frontend/generation_config.json` | `generation_config.json` | default stop ids |
| 4 | `frontend/preprocessor_config.json` | `preprocessor_config.json` | image preprocessing limits and constants |
| 5 | `frontend/video_preprocessor_config.json` | `video_preprocessor_config.json` | video sampling and preprocessing |

The shared Qwen3.6 native Frontend consumes the retained bytes directly and must merge
`tokenizer_config.json::added_tokens_decoder` over the base tokenizer. Parsing only
`tokenizer.json` would truncate the valid domain at id 248069 and is nonconforming. The 35B
artifact retains its own copy of the family resource set so that every artifact remains
self-contained; this duplicate storage does not create a second renderer or processor
implementation. The Python reference may materialize the retained bytes under their source
filenames. MRoPE positions, patch data, visual embeddings, and `rope_delta` are request state rather
than artifact objects.

The 2026-07-18 official-checkpoint cutover rechecked all six hashes and the artifact-native
Transformers frontend. The existing 35B artifact already conformed and was retained unchanged.

## 5. Text and optimized-draft inventory

Every quantized tensor uses `row-split-k128-v1`. Every direct tensor uses `contiguous-le-v1`.

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,2048]` | `W8G32_F16S` |
| after all layers | `text/final_norm` | `[2048]` | `BF16` |
| next | `text/output_head` | `[248320,2048]` | `Q6G64_F16S` |

Embedding and output head are independent. MTP aliases both but does not tie them to one another.

### 5.2 Full-attention Text layer

For each full-attention layer `l` in Section 2, emit these eleven objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[2048]` | `BF16` |
| 1 | `text/layers/{l}/attention/query_key_gate_value` | `[9216,2048]` | `W8G32_F16S` |
| 2 | `text/layers/{l}/attention/query_norm` | `[256]` | `BF16` |
| 3 | `text/layers/{l}/attention/key_norm` | `[256]` | `BF16` |
| 4 | `text/layers/{l}/attention/output` | `[2048,4096]` | `W8G32_F16S` |
| 5 | `text/layers/{l}/post_attention_norm` | `[2048]` | `BF16` |
| 6 | `text/layers/{l}/moe/router_shared_gate` | `[257,2048]` | `BF16` |
| 7 | `text/layers/{l}/moe/routed_gate_up` | `[262144,2048]` | `Q4G64_F16S` |
| 8 | `text/layers/{l}/moe/routed_down` | `[524288,512]` | `Q5G64_F16S` or Section 5.4 Q6 |
| 9 | `text/layers/{l}/moe/shared_gate_up` | `[1024,2048]` | `W8G32_F16S` |
| 10 | `text/layers/{l}/moe/shared_down` | `[2048,512]` | `W8G32_F16S` |

### 5.3 GDN Text layer

For every other layer `l`, emit these fourteen objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[2048]` | `BF16` |
| 1 | `text/layers/{l}/gdn/a_log` | `[32]` | `FP32` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[32]` | `FP32` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,8192]` | `BF16` |
| 4 | `text/layers/{l}/gdn/a_b_projection` | `[64,2048]` | `BF16` |
| 5 | `text/layers/{l}/gdn/query_key_value_z` | `[12288,2048]` | `W8G32_F16S` |
| 6 | `text/layers/{l}/gdn/norm` | `[128]` | `BF16` |
| 7 | `text/layers/{l}/gdn/output` | `[2048,4096]` | `W8G32_F16S` |
| 8 | `text/layers/{l}/post_attention_norm` | `[2048]` | `BF16` |
| 9 | `text/layers/{l}/moe/router_shared_gate` | `[257,2048]` | `BF16` |
| 10 | `text/layers/{l}/moe/routed_gate_up` | `[262144,2048]` | `Q4G64_F16S` |
| 11 | `text/layers/{l}/moe/routed_down` | `[524288,512]` | `Q5G64_F16S` or Section 5.4 Q6 |
| 12 | `text/layers/{l}/moe/shared_gate_up` | `[1024,2048]` | `W8G32_F16S` |
| 13 | `text/layers/{l}/moe/shared_down` | `[2048,512]` | `W8G32_F16S` |

The convolution is stored tap-major. The consumer's channel-major `[8192,4]` view is an explicit
transpose, not a contiguous reshape.

### 5.4 Routed-down precision set

The routed-down format is Q6 exactly for layers:

```text
34, 38, 39
```

It is Q5 for the other 37 layers. The three Q6 objects are full-attention layer 39 and GDN layers
34 and 38.

### 5.5 Optimized draft head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,2048]` | `Q4G64_F16S` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` |

Row `i` represents full-head row `text/draft_head_token_ids[i]`. IDs are unique and lie in
`0..248076`. Target prefill, ordinary target decode, and target verification continue to use the
full output head.

## 6. MTP inventory

The MTP module contains exactly fifteen physical objects:

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `mtp/input_projection` | `[2048,4096]` | `W8G32_F16S` |
| 1 | `mtp/embedding_norm` | `[2048]` | `BF16` |
| 2 | `mtp/hidden_norm` | `[2048]` | `BF16` |
| 3 | `mtp/layer/input_norm` | `[2048]` | `BF16` |
| 4 | `mtp/layer/attention/query_key_gate_value` | `[9216,2048]` | `W8G32_F16S` |
| 5 | `mtp/layer/attention/query_norm` | `[256]` | `BF16` |
| 6 | `mtp/layer/attention/key_norm` | `[256]` | `BF16` |
| 7 | `mtp/layer/attention/output` | `[2048,4096]` | `W8G32_F16S` |
| 8 | `mtp/layer/post_attention_norm` | `[2048]` | `BF16` |
| 9 | `mtp/layer/moe/router_shared_gate` | `[257,2048]` | `BF16` |
| 10 | `mtp/layer/moe/routed_gate_up` | `[262144,2048]` | `W8G32_F16S` |
| 11 | `mtp/layer/moe/routed_down` | `[524288,512]` | `W8G32_F16S` |
| 12 | `mtp/layer/moe/shared_gate_up` | `[1024,2048]` | `W8G32_F16S` |
| 13 | `mtp/layer/moe/shared_down` | `[2048,512]` | `W8G32_F16S` |
| 14 | `mtp/final_norm` | `[2048]` | `BF16` |

MTP token embedding aliases `text/token_embedding`; full proposal/target logits alias
`text/output_head`; the optimized proposal path additionally uses `text/draft_head` and its map.
MTP owns an independent one-layer KV cache at runtime.

## 7. Vision inventory

The Vision policy is the 27B policy with merger output width changed to 2048.

### 7.1 Vision stem

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/patch_embedding` | `[1152,1536]` | `Q6G64_F16S` |
| 1 | `vision/patch_embedding_bias` | `[1152]` | `BF16` |
| 2 | `vision/position_embedding` | `[2304,1152]` | `BF16` |

### 7.2 Vision transformer block

For every block `b` in `0..26`, emit these twelve objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `vision/layers/{b}/attention/qkv` | `[3456,1152]` | `Q4G64_F16S` |
| 1 | `vision/layers/{b}/attention/qkv_bias` | `[3456]` | `BF16` |
| 2 | `vision/layers/{b}/attention/output` | `[1152,1152]` | `Q5G64_F16S` |
| 3 | `vision/layers/{b}/attention/output_bias` | `[1152]` | `BF16` |
| 4 | `vision/layers/{b}/mlp/fc1` | `[4304,1152]` | `Q4G64_F16S` |
| 5 | `vision/layers/{b}/mlp/fc1_bias` | `[4304]` | `BF16` |
| 6 | `vision/layers/{b}/mlp/fc2` | `[1152,4304]` | `Q5G64_F16S` |
| 7 | `vision/layers/{b}/mlp/fc2_bias` | `[1152]` | `BF16` |
| 8 | `vision/layers/{b}/norm1/weight` | `[1152]` | `BF16` |
| 9 | `vision/layers/{b}/norm1/bias` | `[1152]` | `BF16` |
| 10 | `vision/layers/{b}/norm2/weight` | `[1152]` | `BF16` |
| 11 | `vision/layers/{b}/norm2/bias` | `[1152]` | `BF16` |

### 7.3 Vision merger

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/merger/fc1` | `[4608,4608]` | `W8G32_F16S` |
| 1 | `vision/merger/fc1_bias` | `[4608]` | `BF16` |
| 2 | `vision/merger/fc2` | `[2048,4608]` | `W8G32_F16S` |
| 3 | `vision/merger/fc2_bias` | `[2048]` | `BF16` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `BF16` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `BF16` |

No deep-stack object exists. Pixels, frames, patches, grids, positions, and output embeddings are
transient request data.

## 8. Logical views, expert views, and aliases

All views below are compiled binder facts. They do not become extra objects or JSON fields.

### 8.1 Dense fused views

| Parent object | Logical role | Stored row selection | Shape |
|---|---|---|---|
| `*/attention/query_key_gate_value` | query | `[0,4096)` | `[4096,2048]` |
| same | key | `[4096,4608)` | `[512,2048]` |
| same | output gate | `[4608,8704)` | `[4096,2048]` |
| same | value | `[8704,9216)` | `[512,2048]` |
| `text/layers/{l}/gdn/query_key_value_z` | GDN query | `[0,2048)` | `[2048,2048]` |
| same | GDN key | `[2048,4096)` | `[2048,2048]` |
| same | GDN value | `[4096,8192)` | `[4096,2048]` |
| same | GDN output gate Z | `[8192,12288)` | `[4096,2048]` |

The `*` attention pattern applies to Text full-attention layers and the MTP layer.

For `gdn/a_b_projection`, rows `[0,32)` are A and rows `[32,64)` are B. The binder exposes two
contiguous `[32,2048]` BF16 views to the fixed-shape GDN gating projection. For every
`moe/router_shared_gate`, rows `[0,256)` are router rows and row 256 is the shared-expert sigmoid
gate.

For a shared gate/up matrix, row `r` is gate and row `512+r` is up for intermediate coordinate
`r`. This rule applies to Text and MTP.

### 8.2 Routed-expert views

For a routed gate/up parent and expert id `e`:

```text
expert rows = [e*1024,(e+1)*1024)
gate row r  = e*1024 + r
up row r    = e*1024 + 512 + r
```

For routed down:

```text
expert rows = [e*2048,(e+1)*2048)
```

The binder creates a fixed bank descriptor containing the parent plane bases, the exact row and
expert strides in Section 3.4, per-expert row counts, and expert count 256. It does not create 256
independent allocations or pointers in artifact metadata. The MoE operator consumes selected
logical ids and derives the eight expert views from this descriptor.

### 8.3 Aliases

| Logical consumer role | Stored object(s) |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose of `[4,8192]` stored convolution |

MTP KV state remains independent even though its persistent embedding and heads alias Text
objects.

## 9. Inventory and byte proof

### 9.1 Object counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `10 × 11` | 110 |
| GDN Text layers | `30 × 14` | 420 |
| main Text excluding draft | `3 + 110 + 420` | 533 |
| optimized draft head | weight + id map | 2 |
| MTP | fixed inventory | 15 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 × 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| all tensors | `533 + 2 + 15 + 333` | 883 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `883 + 6` | 889 |

### 9.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | Total |
|---|---:|---:|---:|---:|---:|
| `BF16` | 231 | 0 | 8 | 222 | 461 |
| `FP32` | 60 | 0 | 0 | 0 | 60 |
| `I32` | 0 | 1 | 0 | 0 | 1 |
| `Q4G64_F16S` | 40 | 1 | 0 | 54 | 95 |
| `Q5G64_F16S` | 37 | 0 | 0 | 54 | 91 |
| `Q6G64_F16S` | 4 | 0 | 0 | 1 | 5 |
| `W8G32_F16S` | 161 | 0 | 7 | 2 | 170 |
| total | 533 | 2 | 15 | 333 | 883 |

The 522 direct tensors use `contiguous-le-v1`; the 361 quantized tensors use
`row-split-k128-v1`.

### 9.3 Exact tensor bytes

The byte calculation applies the registered layout formula to every shape, including internal
plane padding.

| Component | Tensor bytes | GiB |
|---|---:|---:|
| main Text | 21,038,461,952 | 19.593594551 |
| optimized draft head and map | 143,130,624 | 0.133300781 |
| MTP | 897,934,336 | 0.836266518 |
| Vision | 280,664,992 | 0.261389643 |
| **tensor payload / H2D bytes** | **22,360,191,904** | **20.824551493** |

The same bytes by format are:

| Format | Bytes |
|---|---:|
| `BF16` | 59,482,080 |
| `FP32` | 7,680 |
| `I32` | 524,288 |
| `Q4G64_F16S` | 11,679,339,456 |
| `Q5G64_F16S` | 6,630,296,064 |
| `Q6G64_F16S` | 1,027,840,000 |
| `W8G32_F16S` | 2,962,702,336 |

Materializing all tensors into one 256-byte-aligned device arena adds 15,456 bytes between objects:

```text
device weight arena = 22,360,207,360 bytes = 20.824565887 GiB
```

The six current source resources total 12,833,441 host-retained bytes and are not uploaded to the
weight arena. With resources first in canonical order, the current payload cursor ends at
22,373,040,896 bytes; framing/JSON size is an artifact-instance fact.

### 9.4 MoE residency versus decode traffic

Main Text routed banks occupy exactly:

```text
40 Q4 gate/up banks                11,408,506,880 bytes
37 Q5 + 3 Q6 down banks            7,147,094,016 bytes
all main routed banks             18,555,600,896 bytes = 17.28125 GiB
```

Every expert has the same stride within a bank. Top-8 therefore selects exactly `8/256 = 1/32` of
the routed payload, or 579,862,528 bytes = 553 MiB per target token across 40 layers. This is a
useful-weight byte count, not a measured HBM transaction count. The kernel must not turn it into a
17.28125-GiB/token scan.

## 10. Selected source checkpoint and preflight

The selected conversion source is the official `Qwen/Qwen3.6-35B-A3B` checkpoint pinned to revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`.

The implementation is organized without sibling-target dependencies:

```text
tools/convert/common/
    artifact-independent safetensors reading and numeric quantization

tools/convert/qwen3_6/common/
    Qwen3.6 recipe expressions, shortlist mechanics, shared Vision geometry/mapping,
    frontend resource names, and artifact-writing mechanics

tools/convert/qwen3_6_27b/
    complete 27B target inventory, source mapping, config, ranking policy, and CLI

tools/convert/qwen3_6_35b_a3b/
    complete 35B-A3B target inventory, MoE/dense source mapping, config,
    fixed ranking policy, and CLI
```

Both converter targets depend inward on converter-family common leaves; neither converter imports
the other. Converter-family common code has no target identity, complete inventory, MoE precision
set, or ranking path.

The source checkout path is not artifact metadata or a runtime validity condition. The converter
resolves tensors through `model.safetensors.index.json` and requires these facts
before emitting any payload:

| Configuration field | Required value |
|---|---|
| `architectures` | `["Qwen3_5MoeForConditionalGeneration"]` |
| `model_type` | `"qwen3_5_moe"` |
| `tie_word_embeddings` | `false` |
| `text_config.hidden_size` / `num_hidden_layers` | 2048 / 40 |
| `text_config.vocab_size` | 248320 |
| `text_config.num_attention_heads` / `num_key_value_heads` | 16 / 2 |
| `text_config.head_dim` | 256 |
| `text_config.attn_output_gate` | `true` |
| `text_config.hidden_act` | `"silu"` |
| `text_config.full_attention_interval` | 4 |
| `text_config.linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 |
| `text_config.linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `text_config.linear_conv_kernel_dim` | 4 |
| `text_config.num_experts` / `num_experts_per_tok` | 256 / 8 |
| `text_config.moe_intermediate_size` | 512 |
| `text_config.shared_expert_intermediate_size` | 512 |
| `text_config.tie_word_embeddings` | `false` |
| `text_config.attention_bias` / `attention_dropout` | `false` / `0.0` |
| `text_config.rms_norm_eps` | `1e-6` |
| `text_config.rope_parameters.rope_theta` | 10000000 |
| `text_config.rope_parameters.mrope_section` | `[11,11,10]` |
| `text_config.rope_parameters.mrope_interleaved` | `true` |
| `text_config.rope_parameters.partial_rotary_factor` | `0.25` |
| `text_config.mamba_ssm_dtype` | `"float32"` |
| `text_config.mtp_num_hidden_layers` | 1 |
| `text_config.mtp_use_dedicated_embeddings` | `false` |
| `text_config.max_position_embeddings` | 262144 |
| `vision_config.depth` / `hidden_size` | 27 / 1152 |
| `vision_config.intermediate_size` / `num_heads` | 4304 / 16 |
| `vision_config.temporal_patch_size` / `patch_size` | 2 / 16 |
| `vision_config.spatial_merge_size` | 2 |
| `vision_config.num_position_embeddings` | 2304 |
| `vision_config.out_hidden_size` | 2048 |
| `vision_config.hidden_act` | `"gelu_pytorch_tanh"` |
| `vision_config.deepstack_visual_indexes` | `[]` |
| root Vision token ids | start 248053, end 248054, image 248056, video 248057 |

`text_config.layer_types` must agree with the ten full-attention indices. The source inventory must
contain exactly the 1045 tensors covered by Sections 11 through 13, with the exact shapes and BF16
source dtype:

```text
Text = 3 globals + 10 * 15 full-attention-layer sources
                 + 30 * 18 GDN-layer sources = 693
MTP  = 19
Vision = 333
total = 693 + 19 + 333 = 1045
```

Frontend files must exist under their exact names and are retained byte-for-byte. Before an
artifact writer is opened, the common Qwen3.6 resource preflight requires all six payload SHA-256
values to equal the official family profile recorded in the 27B artifact contract. This guard is
passive source validation only; tokenizer and processor semantics remain owned by the family
frontend rather than duplicated in either target converter.

## 11. Common conversion rules

### 11.1 Direct tensors

- BF16 objects preserve source BF16 words after the stated concatenate, half-split flatten,
  reshape, or transpose.
- GDN `a_log` and `dt_bias` expand their BF16 source values exactly to binary32.
- The draft-head id map is constructed as checked integers and stored as I32.

No other direct dtype conversion is part of this recipe.

### 11.2 Quantized tensors

Every quantized object uses `MAXABS_F16_RECIP_RNE_V1` exactly as defined by the numeric-format
authority:

1. finish the target-specific concatenate/half-split/reshape and obtain a contiguous `[N,K]` BF16
   matrix;
2. expand source words to FP32 for encoder arithmetic;
3. quantize every logical row and every G64 or G32 group independently using the registered
   binary16 scale, reciprocal-multiply, ties-to-even, clamp, code domain, and minimum-subnormal
   rules; and
4. encode the logical codes and scales with `row-split-k128-v1`, including its registered physical
   tail out to `K_pad = align_up(K,128)`.

Groups never cross an output row. Expert and gate/up identities are encoded by whole-row ranges,
so quantization cannot mix them. All Text/MTP K values and all Vision K values except block
`mlp/fc2` are divisible by 128. Vision `mlp/fc2` records logical K=4304 and uses the registered zero
padding to K=4352; the generic layout rule remains the size and decode authority everywhere.

## 12. Text source mapping

Let:

```text
P(l) = model.language_model.layers.{l}.
```

### 12.1 Text globals

| Artifact object | Source | Transform |
|---|---|---|
| `text/token_embedding` | `model.language_model.embed_tokens.weight [248320,2048]` | quantize W8 |
| `text/final_norm` | `model.language_model.norm.weight [2048]` | preserve BF16 |
| `text/output_head` | `lm_head.weight [248320,2048]` | quantize Q6 |

### 12.2 Full-attention source transform

The source q-projection is `[8192,2048]` with per-head `[query_256,gate_256]` rows. Define:

```text
qg    = q_proj.reshape(16,512,2048)
query = qg[:,0:256,:].reshape(4096,2048)
gate  = qg[:,256:512,:].reshape(4096,2048)
qkgv  = concatenate_rows(query, k_proj, gate, v_proj)   # [9216,2048]
```

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight` | preserve BF16 |
| `attention/query_key_gate_value` | `self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight` | form `qkgv`, quantize W8 |
| `attention/query_norm` | `self_attn.q_norm.weight` | preserve BF16 |
| `attention/key_norm` | `self_attn.k_norm.weight` | preserve BF16 |
| `attention/output` | `self_attn.o_proj.weight [2048,4096]` | quantize W8 |
| `post_attention_norm` | `post_attention_layernorm.weight` | preserve BF16 |

### 12.3 GDN source transform

The source QKV row order is query `[0,2048)`, key `[2048,4096)`, and value `[4096,8192)`.

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight` | preserve BF16 |
| `gdn/a_log` | `linear_attn.A_log [32]` | expand to FP32 |
| `gdn/dt_bias` | `linear_attn.dt_bias [32]` | expand to FP32 |
| `gdn/convolution` | `linear_attn.conv1d.weight [8192,1,4]` | take `[:,0,:]`, transpose to `[4,8192]` |
| `gdn/a_b_projection` | `linear_attn.in_proj_a.weight`, `in_proj_b.weight`, each `[32,2048]` | concatenate `[A,B]`, preserve BF16 |
| `gdn/query_key_value_z` | `linear_attn.in_proj_qkv.weight [8192,2048]`, `in_proj_z.weight [4096,2048]` | concatenate `[q,k,v,z]`, quantize W8 |
| `gdn/norm` | `linear_attn.norm.weight [128]` | preserve BF16 |
| `gdn/output` | `linear_attn.out_proj.weight [2048,4096]` | quantize W8 |
| `post_attention_norm` | `post_attention_layernorm.weight` | preserve BF16 |

### 12.4 MoE transform shared by every layer

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `moe/router_shared_gate` | `mlp.gate.weight [256,2048]`, `mlp.shared_expert_gate.weight [1,2048]` | concatenate, preserve BF16 |
| `moe/routed_gate_up` | `mlp.experts.gate_up_proj [256,1024,2048]` | preserve expert-local `[gate half,up half]`, flatten, quantize Q4 |
| `moe/routed_down` | `mlp.experts.down_proj [256,2048,512]` | expert-major flatten, quantize Q5/Q6 by Section 5.4 |
| `moe/shared_gate_up` | shared `gate_proj.weight`, `up_proj.weight`, each `[512,2048]` | concatenate `[gate,up]`, quantize W8 |
| `moe/shared_down` | shared `down_proj.weight [2048,512]` | quantize W8 |

### 12.5 Draft-head construction

The final 35B draft-head id map uses the existing 27B-measured ranking directly:

```text
tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
```

This is a deliberate target-profile decision. Section 2 establishes that the 27B and 35B
token-to-id domains are identical, so the measured ranking has exactly the same interpretation for
both checkpoints. Its existing manifest remains descriptive measurement provenance; the converter
does not require a separately regenerated 35B ranking.

Padded matrix rows `248077..248319` are not selection candidates. Apply the 27B selection algorithm
unchanged over ids `0..248076`:

1. form the ascending forced-id set from entries whose complete merged
   `added_tokens_decoder[id].special` is true;
2. stable-sort all candidate ids by descending row-0 count, which gives ascending-id count ties;
3. take the first `131072 - len(forced)` non-forced ids;
4. append the ascending forced ids and stable-sort that sequence by descending count;
5. require exactly 131072 unique ids, all in `0..248076`;
6. write those ids as I32 in the resulting order; and
7. gather the same ordered rows from the 35B BF16 `lm_head.weight` before Q4 quantization.

The counts were measured from 27B teacher-forced argmax outputs, and that provenance is retained in
the ranking manifest and conversion report. It does not change the fixed 35B storage contract.

For this fixed id map, write the I32 ids, gather
`lm_head.weight[selected_ids,:]` from the 35B source as BF16 `[131072,2048]`, and quantize those
rows Q4. Reusing the 27B `[131072,5120]` draft payload, its full head, its embedding, or any packed
row bytes is nonconforming. Ranking inputs and reports are conversion provenance; only the accepted
selected id map and its 35B-derived weight rows are persistent.

## 13. MTP and Vision source mapping

### 13.1 MTP

Let `M = mtp.layers.0.`. Apply the Text full-attention q/gate extraction and MoE row transforms.

| Artifact object | Source | Transform |
|---|---|---|
| `mtp/input_projection` | `mtp.fc.weight [2048,4096]` | quantize W8 |
| `mtp/embedding_norm` | `mtp.pre_fc_norm_embedding.weight` | preserve BF16 |
| `mtp/hidden_norm` | `mtp.pre_fc_norm_hidden.weight` | preserve BF16 |
| `mtp/layer/input_norm` | `M + input_layernorm.weight` | preserve BF16 |
| `mtp/layer/attention/query_key_gate_value` | `M + self_attn.{q,k,v}_proj.weight` | `[q,k,gate,v]`, quantize W8 |
| `mtp/layer/attention/query_norm` | `M + self_attn.q_norm.weight` | preserve BF16 |
| `mtp/layer/attention/key_norm` | `M + self_attn.k_norm.weight` | preserve BF16 |
| `mtp/layer/attention/output` | `M + self_attn.o_proj.weight` | quantize W8 |
| `mtp/layer/post_attention_norm` | `M + post_attention_layernorm.weight` | preserve BF16 |
| `mtp/layer/moe/router_shared_gate` | `M + mlp.gate.weight`, `M + mlp.shared_expert_gate.weight` | concatenate, preserve BF16 |
| `mtp/layer/moe/routed_gate_up` | `M + mlp.experts.gate_up_proj` | preserve expert-local `[gate half,up half]`, flatten, quantize W8 |
| `mtp/layer/moe/routed_down` | `M + mlp.experts.down_proj` | flatten, quantize W8 |
| `mtp/layer/moe/shared_gate_up` | `M + mlp.shared_expert.gate_proj.weight`, `M + mlp.shared_expert.up_proj.weight` | concatenate `[gate,up]`, quantize W8 |
| `mtp/layer/moe/shared_down` | `M + mlp.shared_expert.down_proj.weight` | quantize W8 |
| `mtp/final_norm` | `mtp.norm.weight` | preserve BF16 |

### 13.2 Vision

All sources begin with `model.visual.`. The stem mapping is:

| Artifact object | Source suffix | Transform |
|---|---|---|
| `vision/patch_embedding` | `patch_embed.proj.weight [1152,3,2,16,16]` | contiguous reshape to `[1152,1536]`, quantize Q6 |
| `vision/patch_embedding_bias` | `patch_embed.proj.bias [1152]` | preserve BF16 |
| `vision/position_embedding` | `pos_embed.weight [2304,1152]` | preserve BF16 |

For block `b`, use source prefix `model.visual.blocks.{b}.`:

| Artifact suffix under `vision/layers/{b}/` | Source suffix | Transform |
|---|---|---|
| `attention/qkv` | `attn.qkv.weight [3456,1152]` | quantize Q4 |
| `attention/qkv_bias` | `attn.qkv.bias [3456]` | preserve BF16 |
| `attention/output` | `attn.proj.weight [1152,1152]` | quantize Q5 |
| `attention/output_bias` | `attn.proj.bias [1152]` | preserve BF16 |
| `mlp/fc1` | `mlp.linear_fc1.weight [4304,1152]` | quantize Q4 |
| `mlp/fc1_bias` | `mlp.linear_fc1.bias [4304]` | preserve BF16 |
| `mlp/fc2` | `mlp.linear_fc2.weight [1152,4304]` | quantize Q5 |
| `mlp/fc2_bias` | `mlp.linear_fc2.bias [1152]` | preserve BF16 |
| `norm1/weight` | `norm1.weight [1152]` | preserve BF16 |
| `norm1/bias` | `norm1.bias [1152]` | preserve BF16 |
| `norm2/weight` | `norm2.weight [1152]` | preserve BF16 |
| `norm2/bias` | `norm2.bias [1152]` | preserve BF16 |

For source prefix `model.visual.merger.`:

| Artifact suffix under `vision/merger/` | Source suffix | Transform |
|---|---|---|
| `fc1` | `linear_fc1.weight [4608,4608]` | quantize W8 |
| `fc1_bias` | `linear_fc1.bias [4608]` | preserve BF16 |
| `fc2` | `linear_fc2.weight [2048,4608]` | quantize W8 |
| `fc2_bias` | `linear_fc2.bias [2048]` | preserve BF16 |
| `norm/weight` | `norm.weight [1152]` | preserve BF16 |
| `norm/bias` | `norm.bias [1152]` | preserve BF16 |

## 14. Binder and materialization obligations

After the common reader establishes version-1 framing and registered layout geometry, the
Qwen3.6-35B-A3B binder:

1. require `model_id == "qwen3.6-35b-a3b"`;
2. generate and consume exactly the 889 objects in Sections 4 through 7, rejecting missing, extra,
   duplicate-role, or alternate-profile objects;
3. require every exact shape, format, layout, and resource encoding, including the Q6 routed-down
   set `{34,38,39}`;
4. validate all 883 tensors, then materialize the startup-selected groups directly into one compact
   256-byte-aligned device arena, with no persistent decoded or repacked copy; the all-features
   profile is 22,360,207,360 bytes;
5. build the full-attention, GDN half-split A/B, router/shared-gate, shared gate/up, and
   routed-expert views in Section 8 once during binding;
6. interpret the object rows in the compiled expert-major order, build descriptors for 256
   equal-stride experts and half-split gate/up rows, and use router id directly as expert id;
   converter and offline verification establish the stored value order;
7. always materialize all 40 Text layers, the full embedding/head, and final norm; materialize the
   validated optimized draft head only when selected at startup;
8. require 131072 unique draft ids in `0..248076` and bind shortlist row `i` to map entry `i`;
9. validate all MTP-specific tensors, materializing their Text aliases and independent runtime KV
   only when the startup draft window is nonzero;
10. validate the complete 27-block Vision tower and 2048-wide merger, materializing them only when
    Vision is enabled at startup;
11. retain and validate all six frontend resources for the loaded-model lifetime; and
12. publish Text plus exactly the MTP, optimized proposal, and Vision groups selected at startup,
    and the Frontend, only after complete artifact validation.

The binder does not know checkpoint paths, safetensors shards, GGUF types, importance matrices,
ranking inputs, quantization operations, or conversion-report fields. The common reader does not
know expert roles or the closed target inventory.

## 15. Decode, prefill, and fusion contract

### 15.1 Sparse-MoE execution

The semantically closed sparse-MoE Op owns router selection, routed experts, gated shared expert,
and their reduction. The qualified repository-internal decode topology is one graph-stable
four-kernel sequence per layer, not eight expert launches or calls through the generic linear
planner:

1. one projection of the 257-row `router_shared_gate` object;
2. one device-side softmax/top-8/normalization plus the independent shared-gate sigmoid;
3. one selected-routed-plus-shared gate/up projection with SwiGLU; and
4. one selected-routed-plus-shared down projection with route reduction, shared scaling, final
   addition, and decoder residual add.

This is a launch/dataflow baseline, not a numerical evaluation order. Each kernel may choose its
natural accumulator, operand staging, intermediate materialization, and workspace representation.

The top-8 stage carries each route weight with its expert id. It may stable-sort those pairs by
expert id before projection to make the eight plane streams monotonic; it must not change logical
selection or detach weights from ids. The Op derives the eight views from the Section 3.4 bank
descriptor on device. No selected expert is gathered into another weight buffer.

The qualified specialized gate/up geometry is one CTA per intermediate coordinate:

```text
grid.x       = 512
block        = 9 warps
warp 0..7    = selected routed slots
warp 8       = shared expert
private act  = [9,512], slot-major; no public gate/up tensor
```

The CTA cooperatively stages the 2048-element input once. Each routed warp reads matching rows
`e*1024+r` and `e*1024+512+r`; the shared warp reads rows `r` and `512+r` from its W8 object. Each
warp computes the gate/up dot products and SwiGLU. Their accumulator and staging types are private
implementation choices.

The corresponding qualified down geometry is one CTA per hidden output coordinate:

```text
grid.x       = 2048
block        = 9 warps
warp 0..7    = selected routed slots
warp 8       = shared expert
private act  = [9,512]
output       = one BF16 routed/shared/residual result; no public expert-output tensor
```

Every routed warp computes one selected expert row and applies its matching route weight. The CTA
combines routed and shared contributions with the decoder residual. Intermediate precision and
reduction association remain private to the implementation. A K=512 row is a dedicated kernel
case: Q5, Q6, and W8 consume exactly 336, 400, and 544 encoded bytes per row. Reusing the current
generic Small-T K=1024 loop would put the complete K=512 row in its scalar tail and is not an
accepted decode path.

This nine-warp decomposition is the qualified decode implementation profile, not a persistent
artifact field. D1 uses 257 8-warp row CTAs, D2 uses one warp, D3 uses the mapping above, and D4
uses the corresponding one-row mapping. W8 D3 assigns one value to each of 32 lanes; D4 retains
paired values per active lane because that is its measured producer-aware winner. The attempted
D1+D2 draining fusion was rejected because caller-owned transient workspace has no initialized
counter state and W8+W8 had no complete-route gain. Per-expert launches, all-expert scans, and
materialized `[8,gate/up,512]` or `[8,2048]` intermediates remain nonconforming.

The closed Op has three private dispatch regimes. `T=1` retains the four-launch decode route. The
Small-T CUDA Core/SIMT implementation admits `T=2..44`: exact-T router templates share each weight
row across all columns, one warp per token performs independent stable top-8 selection, and one
persistent D3 plus one persistent D4 launch advances through the token columns while retaining
disjoint FP32 `[9,512]` activation regions. This is a four-launch route without tensor cores or a
physical expert-job list.

The prefill implementation first projects all router columns with BF16 Tensor Core MMA, selects and
counts assignments on device, prefix-sums per-expert counts, gathers input columns into expert-major
order, and builds flattened expert/column jobs. Routed Q4 gate/up and Q5/Q6 down use persistent BF16
MMA contractions with FP32 accumulation; W8 routed and shared contractions use their exact fixed
shapes. A route-weighted inverse reduction and the shared-down epilogue perform the sole BF16
residual write. No host synchronization, selected-weight repack, capacity factor, or token drop is
introduced.

Production selects the first trace-like RTX 5090 crossover for each routed codec: prefill begins at
`T=45` for Q4+Q5 and Q4+Q6, and `T=18` for W8+W8; smaller multi-column shapes use Small-T.
The prefill route changes from 32-column/four-warp jobs to 64-column/eight-warp jobs at `T=768` and
slices extents larger than 4096 without changing semantics. The workspace query reserves the
largest required decode, Small-T, or at-most-4096-column prefill lifetime, so one caller-owned
graph-stable allocation covers every positive requested maximum.

The logical sparse-MoE formula is:

```text
scores       = Linear(router_shared_gate, X)
p            = softmax(scores[0:256])
ids          = top8(p)
alpha        = p[ids] / sum(p[ids])
routed       = sum_e alpha[e] * Expert_e(X)
shared       = sigmoid(scores[256]) * Shared(X)
residual'    = residual + routed + shared
```

The independent oracle exact-dequantizes artifact codes/scales and evaluates this complete formula
naively in FP32/FP64 from the public inputs, then compares the declared BF16 residual output.
Production routes may use different private rounding and reduction profiles and qualify against the
same oracle with route-appropriate tolerances. Capacity factors, token dropping, persistent expert
permutation, or an all-expert fallback are not allowed.

### 15.2 Other fused projections

The registered 35B target requires fixed-shape decode specializations rather than dispatching these
objects as unrelated generic linear calls:

- `attn_input_decode_w8_9216x2048` reads the one `[q,k,gate,v]` parent once and produces all four
  logical outputs in one launch;
- `gdn_input_decode_w8_12288x2048` reads the one `[q,k,v,z]` parent once, writes Q/K/V in the
  contiguous convolution-input order, and writes Z to its independent consumer;
- `gdn_ab_decode_bf16_64x2048` consumes the two half-split 32-row views in one launch and directly
  produces FP32 `g` and `beta`, including the `A_log`/`dt_bias`, softplus, and sigmoid functions;
- the attention and GDN W8 output projections fuse the decoder residual add; and
- the Q6 full head and Q4 draft head use K=2048 fixed warp-row specializations rather than a
  shape-generic fallback.

The central Op layer implements the two W8 input-projection contracts for every positive target T:
fixed K=2048 decode at `T=1`, closed SIMT/MMA routes for larger T, direct writes to the independent
consumer allocations, and zero transient workspace. The registered Variant uses those Ops inside
its closed projection leaves; the family runtime owns the surrounding schedule. The current
contracts and implementations are in `include/ninfer/ops/` and `src/ops/`.

For `T>1`, the logical Q/K/gate/V or Q/K/V/Z slices of a single `[N,T]` projection do not become
independent contiguous tensors: their token stride remains the parent `N`. The fused input kernel
must therefore accept separate output pointers or explicit output leading dimensions and row
offsets, and scatter each logical output directly. All specified boundaries are multiples of the
64-row MMA tile, so this does not require divergent partial tiles.

The registered row-split codecs match the current operator load model. Small-T kernels issue
coalesced 16-byte plane reads; for each K=1024 slab the encoded streams are:

| Format | Base/code | High | Scale |
|---|---:|---:|---:|
| Q4 | 512 B | 0 | 32 B |
| Q5 | 512 B | 128 B | 32 B |
| Q6 | 512 B | 256 B | 32 B |
| W8 | 1024 B | 0 | 64 B |

The prefill route dequantizes BK=64 tiles to BF16 shared memory and feeds Tensor Core MMA. These
accesses, the half-split gate/up MMA mapping, and the K=512 special case are why this contract keeps
`row-split-k128-v1` instead of defining a MoE-only codec or runtime repack. Its
implementation-private operand staging includes:

```text
mma_weight_stage = BF16(FP32(code) * FP32(binary16_scale))
```

The oracle does not reproduce this staging. Small-T and MMA routes are compared independently with
the same exact-dequantized naive oracle, using tolerances appropriate to their implementation
profiles.

Full attention still preserves the public Q/K norm+MRoPE and post-attention sigmoid Op boundaries.
GDN still preserves convolution, L2 normalization, FP32 recurrent state, gated norm, and residual
Op/state boundaries. Vision retains the implemented 27B block and merger Op boundaries. None of
these boundaries prescribes a private accumulator or intermediate rounding path.

The family schedule composes repository-internal Ops. CUDA implementations remain under
`src/ops`; the 35B package supplies its immutable views, dimensions, graph frontiers, and three
closed leaf families while the family runtime owns layer order, state, graph mechanics, and
workspace composition.

### 15.3 Prefill and MTP

MTP, ordinary generation, and prefill call the same `SparseMoe` API. Their different schedule
shapes do not select different Op semantics. Ordinary generation selects decode. Main Q4+Q5/Q6
verification and rebuild windows use Small-T below their codec crossover, while MTP W8 uses
Small-T through T6 and prefill from T7. Text prefill uses the grouped Tensor Core route. Every path
consumes the same persistent weights directly.

Assignment grouping, job construction, gather, and inverse reduction are private device stages of
the prefill implementation, not semantic or artifact requirements. One host launch per active
expert, selected-weight repacking, capacity factors, and token dropping remain disallowed.

### 15.4 Vision workspace lifetime

Vision attention is segmented and different media items do not attend to one another. The family
runtime therefore encodes media items sequentially with item-bounded GPU workspace instead of
allocating Vision intermediates for every media item in the complete 256K prompt at once. The
merged result is consumed in item order by Text and shifted-MTP prefill.

This lifetime rule does not remove image/video support or change preprocessing. It prevents
independent Vision segments from multiplying peak GPU memory solely because the Text context is
large. The family runtime now applies the same item-bounded GPU lifetime to both Variants while
preserving each item's owning host controls and merged output until its ordered Text/MTP consumers
complete.

## 16. RTX 5090 residency proof

The capacity goal is one user, one active request, one RTX 5090, native context
`C = 262144 = 2^18`, INT8-G64 KV, MTP draft window up to five, CUDA Graphs, and complete resident
Text/MTP/Vision weights. Exact persistent payloads and explicit implementation ceilings both count
against the admission envelope below.

The local device reports 32607 MiB = 31.842773438 GiB through NVIDIA tooling. That device-visible
number, rather than an assumed decimal 32 GB, is the stricter budget used here. Runtime admission
still uses `cudaMemGetInfo` after weight loading so another process cannot invalidate the plan.

### 16.1 INT8 KV

`C` is already a multiple of the current KV layout's 128-token padding. Per full-attention layer
and token:

```text
K/V codes  = 2 planes * 2 KV heads * 256 dimensions            = 1024 bytes
K/V scales = 2 planes * 2 KV heads * (256/64 groups) * 2 bytes =   32 bytes
total                                                             1056 bytes
```

| Cache | Derivation | Bytes | GiB |
|---|---:|---:|---:|
| Text | `10 * C * 1056` | 2,768,240,640 | 2.578125 |
| MTP | `1 * C * 1056` | 276,824,064 | 0.2578125 |
| **total** | `11 * C * 1056` | **3,045,064,704** | **2.8359375** |

The scale bytes are included. There is no hidden page table in the current contiguous physical KV
container. BF16 Text+MTP KV would be 5.5 GiB; it is not the 256K budget proof used here.

### 16.2 GDN snapshots and fixed sequence state

One GDN logical snapshot slot is:

```text
BF16 convolution history = 30 * 8192 * 3 * 2       =  1.40625 MiB
FP32 recurrent matrices  = 30 * 32 * 128 * 128 * 4 = 60.00000 MiB
one slot                                               61.40625 MiB
```

The family state policy fixes `snapshot_slots = mtp_k + 2`. At `mtp_k=5`, seven slots consume
450,723,840 bytes = 429.84375 MiB. Applying the family Program's fixed state layout to the 35B
widths and a six-column verification window gives an 8,207,872-byte ceiling for fixed
token/logit/hidden/ledger/sampling buffers. The algorithm is shared; the resulting byte count is a
35B configuration result rather than a checkpoint payload property.

```text
planned sequence-persistent reservation
  = INT8 Text+MTP KV + seven GDN slots + fixed state
  = 3,503,996,416 bytes
  = 3.263350964 GiB
```

### 16.3 Workspace and final envelope

The family planner dry-runs the same scoped allocations used by Text, MTP, and Vision and reserves
their maximum, not their sum. For the 35B native configuration with prefill chunk 1024, the
item-bounded Vision maximum (`P=131072`, `V=32768`, up to 384 segments) dominates and produces one
2,039,482,112-byte stable workspace arena. Text/MTP and each Vision item reuse that arena; neither a
full-bank dequantization nor aggregate intermediates for every media item are resident.

| Resident or reserved item | Bytes | MiB |
|---|---:|---:|
| device weight arena | 22,360,207,360 | 21,324.355469 |
| complete sequence-persistent arena | 3,503,996,416 | 3,341.671387 |
| shared Text/MTP/Vision workspace arena | 2,039,482,112 | 1,945.001709 |
| CUDA Graph allowance | 595,591,168 | 568.000000 |
| CUDA context/allocator guard | 1,073,741,824 | 1,024.000000 |
| **planned total** | **29,573,018,880** | **28,203.028564** |
| **remaining from 32607 MiB** | **4,617,898,752** | **4,403.971436** |

The remaining 4.300753355 GiB is deliberate unassigned margin, not permission for another weight
copy.

The workspace and CUDA Graph rows are runtime planning values rather than persistent artifact
bytes. The graph value follows the implemented family calculation for the 35B native configuration:
sixteen ordinary/aligned executables at 12 MiB, four short-window MTP executables at 12 MiB, and four
long-window MTP executables at 82 MiB. Runtime admission uses the exact sequence/workspace plans and
the calculated graph allowance after loading weights; the final guard retains conservative room
for the CUDA context and allocator.

For context, making both gate/up and down banks of every main routed expert Q5 would raise the
weight arena to about 23.230816 GiB and leave 2.786106586 GiB under the same complete envelope. Making
both banks Q6 would raise weights to 26.980816 GiB and exceed that envelope by 0.963893414 GiB. All
W8 would raise weights to 35.418316 GiB and fail before KV. The chosen mixed tiers therefore
materially improve memory margin and selected-expert traffic while retaining extra precision in the
down banks; all-Q5 is not excluded by capacity alone.

## 17. Implementation locations and checks

The registered artifact route is supported by focused evidence tied to this contract:

- conversion/inventory checks establish exactly 883 tensors, six frontend resources, the object
  signatures in this document, and the direct materialized device arena of 22,360,207,360 bytes;
- independent mathematical or exact oracles remain authoritative for numeric codecs, fused
  projections, sparse MoE, attention, GDN, MTP, and Vision Ops; each production numerical path is
  checked against its own oracle and is not required to reproduce another path's output;
- the real public Engine route loads the named artifact as
  `qwen3_6_35b_a3b` and exercises Text, MTP, prefix/state lifecycle, CUDA Graph execution,
  and inline Vision; and
- construction at native `max_context=262144` with INT8-G64 KV and MTP window five verifies the
  admitted sequence/workspace layout and rejects an over-capacity request before Program mutation.

`SparseMoe` is checked directly against its complete independent oracle across the decode,
codec-specific Small-T/prefill transitions, all three routed codec profiles, the 768-column wide
plan, and 4097-column sliced Graph replay. On the RTX 5090 trace-like workload, the complete
Q4+Q5 Op at the primary `T=1024` point takes about 813 microseconds and sustains 72.6 useful
TFLOP/s; larger qualified Q4+Q5 points reach about 105 useful TFLOP/s at T=2048 and 91.9 at T=4096.
These are complete-Op measurements, not a claim that the logical sparse work equals dense executed
FLOPs.

Artifact-native Python execution remains a diagnostic implementation. It does not define an exact
generated-token golden for the C++ runtime, and comparisons between different quantization,
schedule, eager/graph, or reference paths are not acceptance tests.
