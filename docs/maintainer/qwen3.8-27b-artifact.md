# Qwen3.8-27B artifact reference

This reference defines the `qwen3.8-27b/nvfp4` `.ninfer` storage contract: identity, object
inventory, shapes, numeric formats, storage layouts, fused row order, aliases, fixed sources, and
source-to-object transforms. The existing registered `qwen3.8-27b/groupwise-int` contract remains
defined in Section 13.

The NVFP4 profile is a registered Engine identity implemented by the target converter, exact
binder, and Qwen3.8 execution leaves. The generic artifact registry resolves its version-2
identity without a runtime profile flag. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics and state behavior in
[`qwen3.6-27b-model.md`](qwen3.6-27b-model.md).

## 1. NVFP4 artifact identity and contents

```text
filename   = qwen3_8_27b_nvfp4.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b_nvfp4-v1
```

The artifact is one complete image containing Text, the optimized MTP draft head, MTP, Vision, and
six frontend resources. These components are not separate artifacts or selectable storage
profiles. A runtime may choose not to materialize a supported component, but that does not change
the artifact inventory or identity.

The identity is read from the version-2 artifact directory. The filename, object count, and any
representative tensor descriptor do not select the model or weights profile.

## 2. Fixed target facts

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. NVFP4 groups and all
groupwise integer formats quantize along `K`; row-scaled FP8 owns one scale per `N` row.

| Fact | Value |
|---|---:|
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| Text hidden width | 5120 |
| Text layers | 64 |
| Text MLP intermediate width | 17408 |
| full-attention layers | 16 |
| GDN layers | 48 |
| query heads / KV heads / head width | 24 / 4 / 256 |
| query / KV widths | 6144 / 1024 |
| GDN key heads x width | 16 x 128 = 2048 |
| GDN value heads x width | 48 x 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| MTP layers | 1 full-attention dense-MLP layer |
| optimized draft-head rows | 131072 |
| Vision depth / hidden / intermediate width | 27 / 1152 / 4304 |
| Vision heads / patch input width | 16 / 1536 |
| Vision position rows | 2304 |
| Vision merger input / output | 4608 / 5120 |

Full-attention Text layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every other Text layer is GDN. Layer and Vision-block numbers in object names are unpadded decimal
integers.

## 3. Numeric assignment and physical row order

### 3.1 Complete format assignment

The Text backbone preserves the fixed mixed-precision allocation of the quantized Text source. The
embedding is the only additional row-scaled FP8 quantization performed by NInfer.

| Role | Layer domain | Format | Layout | Value provenance |
|---|---|---|---|---|
| token embedding | global | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | encode official BF16 source |
| full-attention input projection | all full-attention layers | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| full-attention output projection | all full-attention layers | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| GDN input projection | all GDN layers | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| GDN output projection | all GDN layers | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| MLP gate/up and down | `0..55` | `NVFP4` | `blockscale-k16-m128x4-v1` | preserve source NVFP4 words |
| MLP gate/up and down | `56..63` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| full output head | global | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` | preserve source FP8 words |
| Text norms, GDN convolution, and fused GDN A/B projection | applicable layers | `BF16` | `contiguous-le-v1` | preserve quantized-source BF16 words |
| GDN `A_log` and `dt_bias` | all GDN layers | `FP32` | `contiguous-le-v1` | expand quantized-source BF16 values |
| NVFP4 input divisors | `0..55` MLP sites | `FP32` | `contiguous-le-v1` | preserve source FP32 words |
| optimized MTP draft head | global | `Q4G64_F16S` | `row-split-k128-v1` | encode official BF16 head rows |
| optimized draft-head id map | global | `I32` | `contiguous-le-v1` | derived index tensor |
| MTP matrices | MTP | `W8G32_F16S` | `row-split-k128-v1` | encode official BF16 source |
| MTP norms | MTP | `BF16` | `contiguous-le-v1` | preserve official BF16 words |
| Vision block input/expansion matrices | Vision | `Q4G64_F16S` | `row-split-k128-v1` | encode official BF16 source |
| Vision block output/contraction matrices | Vision | `Q5G64_F16S` | `row-split-k128-v1` | encode official BF16 source |
| Vision patch projection | Vision | `Q6G64_F16S` | `row-split-k128-v1` | encode official BF16 source |
| Vision merger matrices | Vision | `W8G32_F16S` | `row-split-k128-v1` | encode official BF16 source |
| all other Vision weights and biases | Vision | `BF16` | `contiguous-le-v1` | preserve official BF16 words |

The selected quantized source contains 168 NVFP4 MLP matrices and 233 row-scaled FP8 matrices.
Fusing matrices at the execution-consumer boundary produces 112 NVFP4 parents and 145 FP8 parents.
The locally encoded embedding brings the artifact FP8-parent count to 146. Fusion never decodes or
requantizes a source code or scale word.

### 3.2 Fused parent row order

The artifact parent boundary follows the execution-consumer boundary. Matrices are concatenated
along output rows when they consume the same represented activation at the same semantic rounding
boundary, use the same numeric format and scale semantics, and have no intervening normalization,
nonlinearity, attention, recurrent-state transition, or residual update. The fused families below
are the exhaustive matrix-row fusions in this profile.

The following concatenations define physical output-row order:

- Text full-attention input: `[query,key,output_gate,value]`;
- Text GDN input: `[query,key,value,z]`;
- Text GDN control projection: `[A,B]`;
- Text and MTP MLP input: `[gate,up]`;
- MTP full-attention input: `[query,key,output_gate,value]`.

Within every source full-attention q-projection, each of the 24 heads stores
`[query_256,output_gate_256]`. The converter separates those per-head halves before constructing the
fused parent. For a row-scaled FP8 source, every row move applies identically to its E4M3FN code row
and its BF16 scale word.

The source GDN `in_proj_qkv` row order is `[query 2048,key 2048,value 6144]`; the converter appends
the 6144 `z` rows. The two BF16 control projections are one `a_b_projection [96,5120]` parent with
48 A rows followed by 48 B rows. Every stored GDN convolution is tap-major `[4,10240]`.

The following boundaries are intentionally not physical matrix fusions:

- attention and GDN output projections, MLP down projections, Vision attention outputs, Vision
  contractions, and merger stages consume results produced after an intervening semantic stage;
- the BF16 GDN A/B parent remains separate from the row-scaled FP8 Q/K/V/Z parent because their
  formats differ and the control branch and explicit BF16 projection input are distinct semantic
  rounding domains; the convolution remains a separate non-matrix operand;
- the MTP input projection and every Vision QKV projection are already single source matrices and
  require no row-concatenation transform;
- biases, normalization vectors, GDN parameter vectors, and NVFP4 divisors are not matrix rows;
- token embedding and output head are untied independent matrices and cannot alias or fuse.

### 3.3 Scale ownership

Each `FP8_E4M3FN_ROW_BF16S` object is one composite weight containing a row-major E4M3FN code plane
and one BF16 multiplier per logical row. Source `.weight` and `.weight_scale [N,1]` fields therefore
become one artifact object; the source scale is not an independently named artifact tensor.

Each `NVFP4` parent contains its packed E2M1 code plane, swizzled E4M3FN K16 scale plane, and one
trailing FP32 weight divisor `d_w`. Every NVFP4 parent has one separate rank-zero FP32 input divisor
`d_x`, inserted immediately after the parent. The two objects are bound as one calibrated execution
site; neither divisor may be inferred, defaulted to `1`, or folded into rewritten scale words.

For every layer `0..55`, source gate and up projections must have bit-identical `d_w` words and
bit-identical `d_x` words before they can form one `mlp/gate_up` parent. The down projection supplies
its own pair. This yields exactly 112 input-divisor objects.

## 4. Object namespace, order, and frontend resources

### 4.1 Namespace and writer order

- `text/` contains the embedding, 64 Text layers, final norm, full output head, and optimized MTP
  draft head.
- `mtp/` contains MTP-private tensors.
- `vision/` contains the Vision tower and merger.
- `frontend/` contains the six raw frontend resources.

Objects are written in this order:

1. the six frontend resources in Section 4.2;
2. `text/token_embedding`;
3. Text layers `0..63`, using the applicable object order in Sections 5.2 through 5.4;
4. `text/final_norm` and `text/output_head`;
5. `text/draft_head` and `text/draft_head_token_ids`;
6. the twelve MTP tensors in Section 6;
7. the Vision stem, blocks `0..26`, and merger in Section 7.

Readers bind by name. Object names contain no format spelling. Logical row views and aliases are
not artifact objects or directory records.

### 4.2 Frontend resources

The artifact preserves exactly these official-source files as `raw-bytes-v1` resources:

| Order | Object name | Source filename | Meaning |
|---:|---|---|---|
| 0 | `frontend/tokenizer.json` | `tokenizer.json` | base BPE vocabulary, merges, token bytes, and its added-token subset |
| 1 | `frontend/tokenizer_config.json` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `frontend/chat_template.jinja` | `chat_template.jinja` | registered Qwen template |
| 3 | `frontend/generation_config.json` | `generation_config.json` | default stop ids |
| 4 | `frontend/preprocessor_config.json` | `preprocessor_config.json` | image preprocessing limits and constants |
| 5 | `frontend/video_preprocessor_config.json` | `video_preprocessor_config.json` | video sampling and preprocessing |

## 5. Text and optimized MTP draft inventory

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,5120]` | `FP8_E4M3FN_ROW_BF16S` |
| after all layers | `text/final_norm` | `[5120]` | `BF16` |
| next | `text/output_head` | `[248320,5120]` | `FP8_E4M3FN_ROW_BF16S` |

The embedding and output head are independent objects with different encoder provenance.

### 5.2 Full-attention Text layer

For every full-attention layer `l` in Section 2, emit these six objects before the MLP tail:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/attention/query_key_gate_value` | `[14336,5120]` | `FP8_E4M3FN_ROW_BF16S` |
| 2 | `text/layers/{l}/attention/query_norm` | `[256]` | `BF16` |
| 3 | `text/layers/{l}/attention/key_norm` | `[256]` | `BF16` |
| 4 | `text/layers/{l}/attention/output` | `[5120,6144]` | `FP8_E4M3FN_ROW_BF16S` |
| 5 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |

Append the MLP tail from Section 5.4. A full-attention layer in `0..55` therefore has ten physical
objects; layers 59 and 63 each have eight.

### 5.3 GDN Text layer

For every other layer `l` in `0..63`, emit these nine objects before the MLP tail:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/gdn/a_log` | `[48]` | `FP32` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[48]` | `FP32` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,10240]` | `BF16` |
| 4 | `text/layers/{l}/gdn/a_b_projection` | `[96,5120]` | `BF16` |
| 5 | `text/layers/{l}/gdn/query_key_value_z` | `[16384,5120]` | `FP8_E4M3FN_ROW_BF16S` |
| 6 | `text/layers/{l}/gdn/norm` | `[128]` | `BF16` |
| 7 | `text/layers/{l}/gdn/output` | `[5120,6144]` | `FP8_E4M3FN_ROW_BF16S` |
| 8 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |

Append the MLP tail from Section 5.4. A GDN layer in `0..55` therefore has thirteen physical
objects; each GDN layer in `56..63` has eleven.

### 5.4 Per-layer MLP tail

For layers `0..55`, append these four objects in order:

| Order within tail | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `NVFP4` |
| 1 | `text/layers/{l}/mlp/gate_up_projection/input_scale_divisor` | `[]` | `FP32` |
| 2 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `NVFP4` |
| 3 | `text/layers/{l}/mlp/down_projection/input_scale_divisor` | `[]` | `FP32` |

For layers `56..63`, append these two objects in order:

| Order within tail | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `FP8_E4M3FN_ROW_BF16S` |
| 1 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `FP8_E4M3FN_ROW_BF16S` |

### 5.5 Optimized MTP draft head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,5120]` | `Q4G64_F16S` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` |

Row `i` of `text/draft_head` represents full-head row `text/draft_head_token_ids[i]`. The ids are
unique and lie in `0..248076`.

## 6. MTP inventory

The MTP module contains exactly twelve physical objects:

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `mtp/input_projection` | `[5120,10240]` | `W8G32_F16S` |
| 1 | `mtp/embedding_norm` | `[5120]` | `BF16` |
| 2 | `mtp/hidden_norm` | `[5120]` | `BF16` |
| 3 | `mtp/layer/input_norm` | `[5120]` | `BF16` |
| 4 | `mtp/layer/attention/query_key_gate_value` | `[14336,5120]` | `W8G32_F16S` |
| 5 | `mtp/layer/attention/query_norm` | `[256]` | `BF16` |
| 6 | `mtp/layer/attention/key_norm` | `[256]` | `BF16` |
| 7 | `mtp/layer/attention/output` | `[5120,6144]` | `W8G32_F16S` |
| 8 | `mtp/layer/post_attention_norm` | `[5120]` | `BF16` |
| 9 | `mtp/layer/mlp/gate_up` | `[34816,5120]` | `W8G32_F16S` |
| 10 | `mtp/layer/mlp/down` | `[5120,17408]` | `W8G32_F16S` |
| 11 | `mtp/final_norm` | `[5120]` | `BF16` |

MTP token embedding, full output head, and optimized proposal head are aliases in Section 8.2.

## 7. Vision inventory

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
| 2 | `vision/merger/fc2` | `[5120,4608]` | `W8G32_F16S` |
| 3 | `vision/merger/fc2_bias` | `[5120]` | `BF16` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `BF16` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `BF16` |

No Vision deep-stack object exists.

## 8. Logical views and aliases

All entries in this section are views or aliases of physical objects above. They are not extra
artifact objects.

### 8.1 Fused row views

| Parent object | Logical role | Stored row selection | Shape |
|---|---|---|---|
| `text/layers/{l}/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `text/layers/{l}/gdn/query_key_value_z` | GDN query | `[0,2048)` | `[2048,5120]` |
| same | GDN key | `[2048,4096)` | `[2048,5120]` |
| same | GDN value | `[4096,10240)` | `[6144,5120]` |
| same | GDN z | `[10240,16384)` | `[6144,5120]` |
| `text/layers/{l}/gdn/a_b_projection` | GDN A projection | `[0,48)` | `[48,5120]` |
| same | GDN B projection | `[48,96)` | `[48,5120]` |
| `text/layers/{l}/mlp/gate_up` | MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MLP up | `[17408,34816)` | `[17408,5120]` |
| `mtp/layer/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `mtp/layer/mlp/gate_up` | MTP MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MTP MLP up | `[17408,34816)` | `[17408,5120]` |

`mtp/input_projection [5120,10240]` is a single input-column parent rather than a row-fused parent.
Columns `[0,5120)` multiply the normalized token embedding and columns `[5120,10240)` multiply the
normalized hidden state. A consumer may evaluate those two column domains and accumulate into the
same output without materializing their concatenated BF16 input.

### 8.2 Aliases

| Logical consumer role | Stored object or view |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose view of the stored `[4,10240]` convolution |

### 8.3 Binding and execution-consumer boundary

Every fused matrix parent in Sections 3.2 and 8.1 is one indivisible artifact binding unit and one
complete immutable runtime `Weight`. In particular, this identity binds Text and MTP
`query_key_gate_value`, Text `query_key_value_z`, Text `a_b_projection`, Text and MTP `gate_up`, and
Vision `qkv` as single-parent payloads. The binder and primary projection Ops must consume those
complete parents; they must not materialize their logical row ranges as independent persistent
weights or introduce a split-payload alternative for this identity.

Logical row views exist for indexing, verification, and schedules that intentionally evaluate only
a subset of an already-bound parent. Such a schedule may derive a zero-copy view, but the view does
not change parent ownership or replace the complete-parent Op boundary. A complete-parent
projection may write its independently laid-out logical outputs directly and need not materialize a
packed output tensor.

Separate artifact objects do not by themselves require separate semantic Ops or kernel launches.
Biases, normalization vectors, GDN parameters, convolution, and the differently formatted GDN
parents may remain separate immutable arguments to a larger fused Op whenever that Op preserves the
model's semantic boundaries.

## 9. Inventory summary

### 9.1 Object counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `14 x 10 + 2 x 8` | 156 |
| GDN Text layers | `42 x 13 + 6 x 11` | 612 |
| main Text excluding draft | `3 + 156 + 612` | 771 |
| optimized MTP draft head | weight + id map | 2 |
| MTP | fixed inventory | 12 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 x 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| all tensors | `771 + 2 + 12 + 333` | 1118 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `1118 + 6` | 1124 |

### 9.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | Total |
|---|---:|---:|---:|---:|---:|
| `BF16` | 305 | 0 | 7 | 222 | 534 |
| `FP32` | 208 | 0 | 0 | 0 | 208 |
| `I32` | 0 | 1 | 0 | 0 | 1 |
| `Q4G64_F16S` | 0 | 1 | 0 | 54 | 55 |
| `Q5G64_F16S` | 0 | 0 | 0 | 54 | 54 |
| `Q6G64_F16S` | 0 | 0 | 0 | 1 | 1 |
| `W8G32_F16S` | 0 | 0 | 5 | 2 | 7 |
| `NVFP4` | 112 | 0 | 0 | 0 | 112 |
| `FP8_E4M3FN_ROW_BF16S` | 146 | 0 | 0 | 0 | 146 |
| total | 771 | 2 | 12 | 333 | 1118 |

The artifact contains 743 direct tensors using `contiguous-le-v1`, 117 grouped integer tensors
using `row-split-k128-v1`, 112 NVFP4 tensors using `blockscale-k16-m128x4-v1`, and 146 row-scaled
FP8 tensors using `row-scale-v1`.

## 10. Fixed sources and numeric conversion

### 10.1 Source identities and ownership

The official base source is `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The quantized Text source is
`unsloth/Qwen3.8-27B-NVFP4` revision
`60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`.

| Artifact content | Materialization source |
|---|---|
| Text FP8 and NVFP4 matrices | fixed quantized Text source |
| Text norms, GDN convolution, fused GDN A/B projection, `A_log`, and `dt_bias` | fixed quantized Text source |
| token embedding | official BF16 source |
| optimized draft head and id map | official BF16 source plus fixed ranking input |
| MTP | official BF16 source |
| Vision | official BF16 source |
| six frontend resources | official source |

The quantized source's Vision tensors, MTP tensors, frontend files, and BF16 embedding are not
materialization inputs. Its full-attention `k_scale` and `v_scale` fields are also excluded: they
are not fields of a persistent weight format and do not become artifact tensors. Runtime activation,
KV-cache, and recurrent-state codecs remain outside this artifact contract.

### 10.2 Direct tensors

- Artifact `BF16` objects preserve the selected source BF16 words after the stated concatenate,
  reshape, or transpose.
- GDN `A_log` and `dt_bias` expand selected-source BF16 values exactly to binary32 and store the
  resulting FP32 words.
- Native NVFP4 input-divisor FP32 words are preserved exactly while changing source shape `[1]` to
  artifact shape `[]`.
- `text/draft_head_token_ids` is stored as `I32`.

No other source BF16 field is promoted to FP32.

### 10.3 Preserved row-scaled FP8

For every source-derived FP8 matrix, the converter validates an E4M3FN `.weight [N,K]` and a BF16
`.weight_scale [N,1]`. It preserves all code and scale words exactly. Splitting, row permutation,
and concatenation operate on `(code row, scale word)` pairs. The resulting logical words are encoded
with `row-scale-v1`; no floating-point decode, scale recomputation, or requantization is permitted.

### 10.4 Embedding FP8 encoder

Only `text/token_embedding` uses the target-specific encoder profile
`MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`. The input is the official BF16 matrix. Every BF16 word expands
exactly to binary32, and each row is encoded independently. All operations below use
round-to-nearest, ties-to-even, with no flush-to-zero:

```text
amax = max_k(abs(x[k]))

if amax == +0:
    scale_bf16 = bfloat16(+0)
    code[k]    = E4M3FN(+0) for every k
else:
    raw_scale32 = round_f32(amax / binary32(448))
    scale_bf16  = round_bf16(raw_scale32)

    if scale_bf16 == +0:
        scale_bf16 = bfloat16_from_bits(0x0001)

    scale32 = exact_bfloat16_to_binary32(scale_bf16)
    inv32   = round_f32(binary32(1) / scale32)

    for every k:
        normalized32 = round_f32(x[k] * inv32)
        bounded32     = clamp(normalized32, -448, 448)
        code[k]       = round_e4m3fn_rne(bounded32)
```

The producer rejects a non-finite source value or a non-positive/non-finite scale for a nonzero
row. This encoder profile applies only to the embedding; it makes no claim about how the upstream
FP8 source selected its words.

### 10.5 Preserved NVFP4

For each source NVFP4 matrix, the converter validates:

```text
weight_packed       U8        [N,K/2]
weight_scale        E4M3FN    [N,K/16]
weight_global_scale FP32      [1]
input_global_scale  FP32      [1]
```

The source `weight_global_scale` is the positive divisor `d_w` in the registered NVFP4 decode
formula, and `input_global_scale` is the positive site divisor `d_x`. The converter copies packed
E2M1 words without decoding, permutes the natural scale matrix into
`blockscale-k16-m128x4-v1`, and preserves both FP32 words exactly. Gate/up fusion concatenates rows
only after the equality checks in Section 3.3.

### 10.6 Grouped integer conversion

Draft-head, MTP, and Vision matrices first complete the specified split, concatenate, reshape, or
transpose as one contiguous BF16 logical matrix. They then apply
`MAXABS_F16_RECIP_RNE_V1` independently to every output row and G32 or G64 group along `K`, and use
`row-split-k128-v1`. Groups never cross output rows.

## 11. Text source mapping

Let:

```text
P(l) = model.language_model.layers.{l}.
```

### 11.1 Text globals

| Artifact object | Source | Transform |
|---|---|---|
| `text/token_embedding` | official `model.language_model.embed_tokens.weight [248320,5120]` BF16 | apply Section 10.4 |
| `text/final_norm` | quantized-source `model.language_model.norm.weight [5120]` BF16 | preserve words |
| `text/output_head` | quantized-source `lm_head.weight [248320,5120]` E4M3FN and `weight_scale [248320,1]` BF16 | preserve words |

### 11.2 Full-attention source transform

The quantized source q-projection is `[12288,5120]` with per-head
`[query_256,output_gate_256]` rows:

```text
qg    = q_proj.reshape(24,512,5120)
query = qg[:,0:256,:].reshape(6144,5120)
gate  = qg[:,256:512,:].reshape(6144,5120)
```

The same reshape and selection apply to its `[12288,1]` row-scale tensor.

| Artifact suffix under `text/layers/{l}/` | Quantized source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` BF16 | preserve words |
| `attention/query_key_gate_value` | `self_attn.{q,k,v}_proj.weight` and their row scales | form `[query,key,output_gate,value]`, preserve FP8 words |
| `attention/query_norm` | `self_attn.q_norm.weight [256]` BF16 | preserve words |
| `attention/key_norm` | `self_attn.k_norm.weight [256]` BF16 | preserve words |
| `attention/output` | `self_attn.o_proj.weight [5120,6144]` and row scales | preserve FP8 words |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` BF16 | preserve words |

The MLP tail follows Section 11.4.

### 11.3 GDN source transform

The quantized source `linear_attn.in_proj_qkv.weight [10240,5120]` row order is:

```text
query = [0,2048)
key   = [2048,4096)
value = [4096,10240)
```

| Artifact suffix under `text/layers/{l}/` | Quantized source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` BF16 | preserve words |
| `gdn/a_log` | `linear_attn.A_log [48]` BF16 | exact-expand to FP32 |
| `gdn/dt_bias` | `linear_attn.dt_bias [48]` BF16 | exact-expand to FP32 |
| `gdn/convolution` | `linear_attn.conv1d.weight [10240,1,4]` BF16 | take `[:,0,:]`, transpose to `[4,10240]` |
| `gdn/a_b_projection` | `linear_attn.in_proj_a.weight`, `in_proj_b.weight`, each `[48,5120]` BF16 | concatenate `[A,B]`, preserve words |
| `gdn/query_key_value_z` | `linear_attn.in_proj_qkv.weight`, `in_proj_z.weight [6144,5120]`, and their row scales | form `[query,key,value,z]`, preserve FP8 words |
| `gdn/norm` | `linear_attn.norm.weight [128]` BF16 | preserve words |
| `gdn/output` | `linear_attn.out_proj.weight [5120,6144]` and row scales | preserve FP8 words |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` BF16 | preserve words |

The MLP tail follows Section 11.4.

### 11.4 MLP source transform

For every layer `0..55`, source `mlp.gate_proj` and `mlp.up_proj` each have logical shape
`[17408,5120]`. Concatenate their packed-code and natural scale rows as `[gate,up]` to form
`mlp/gate_up [34816,5120]`; preserve their shared `d_w` in the parent and shared `d_x` in the
following scalar. Map `mlp.down_proj [5120,17408]` to `mlp/down` and its following scalar without a
row transform.

For every layer `56..63`, perform the same `[gate,up]` row concatenation on the source E4M3FN code
rows and BF16 row scales. Map the row-scaled FP8 down projection without a row transform. These
layers have no persistent input-divisor object.

### 11.5 Optimized MTP draft-head construction

The fixed ranking source is:

```text
tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
```

Interpret the file as a little-endian I64 array with row width 248320 and use row 0. Padded rows
`248077..248319` are not candidates. Select ids from `0..248076` as follows:

1. form the ascending forced-id set from entries whose merged
   `added_tokens_decoder[id].special` value is true;
2. stable-sort all candidate ids by descending row-0 count, with ascending-id count ties;
3. take the first `131072 - len(forced)` non-forced ids;
4. append the ascending forced ids and stable-sort the result by descending count;
5. require exactly 131072 unique ids in `0..248076`;
6. store those ids as `I32` in `text/draft_head_token_ids`;
7. gather the same ordered BF16 rows from the official `lm_head.weight` and quantize them to Q4.

## 12. MTP, Vision, and conformance

### 12.1 MTP source mapping

Let `M = mtp.layers.0.` in the official source. Apply the full-attention q/gate extraction from
Section 11.2.

| Artifact object | Official source | Transform |
|---|---|---|
| `mtp/input_projection` | `mtp.fc.weight [5120,10240]` | quantize W8 |
| `mtp/embedding_norm` | `mtp.pre_fc_norm_embedding.weight [5120]` | preserve BF16 |
| `mtp/hidden_norm` | `mtp.pre_fc_norm_hidden.weight [5120]` | preserve BF16 |
| `mtp/layer/input_norm` | `M + input_layernorm.weight [5120]` | preserve BF16 |
| `mtp/layer/attention/query_key_gate_value` | `M + self_attn.{q,k,v}_proj.weight` | form `[query,key,output_gate,value]`, quantize W8 |
| `mtp/layer/attention/query_norm` | `M + self_attn.q_norm.weight [256]` | preserve BF16 |
| `mtp/layer/attention/key_norm` | `M + self_attn.k_norm.weight [256]` | preserve BF16 |
| `mtp/layer/attention/output` | `M + self_attn.o_proj.weight [5120,6144]` | quantize W8 |
| `mtp/layer/post_attention_norm` | `M + post_attention_layernorm.weight [5120]` | preserve BF16 |
| `mtp/layer/mlp/gate_up` | `M + mlp.gate_proj.weight`, `up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize W8 |
| `mtp/layer/mlp/down` | `M + mlp.down_proj.weight [5120,17408]` | quantize W8 |
| `mtp/final_norm` | `mtp.norm.weight [5120]` | preserve BF16 |

### 12.2 Vision source mapping

All sources in this section begin with official-source `model.visual.`.

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
| `fc2` | `linear_fc2.weight [5120,4608]` | quantize W8 |
| `fc2_bias` | `linear_fc2.bias [5120]` | preserve BF16 |
| `norm/weight` | `norm.weight [1152]` | preserve BF16 |
| `norm/bias` | `norm.bias [1152]` | preserve BF16 |

### 12.3 Producer requirements

Before opening the output, the converter must validate both fixed checkpoint configurations, every
selected source name, shape, dtype, format assignment, source scale geometry, frontend resource,
ranking input, and the complete ordered object plan.

During materialization, the converter must:

- preserve every source-derived E4M3FN code and BF16 row-scale word exactly after the defined
  split, permutation, and fusion;
- preserve every NVFP4 packed code, natural scale word, `d_w`, and `d_x` exactly, including
  all gate/up equality requirements;
- preserve direct words exactly and perform the specified BF16-to-FP32 expansions;
- encode the embedding with the bit-level profile in Section 10.4;
- encode draft-head, MTP, and Vision weights with `MAXABS_F16_RECIP_RNE_V1`;
- write the complete inventory, formats, layouts, logical views, aliases, and six resource
  payloads.

Validation must reject an incomplete or alternate mixed-precision allocation. It must not fill a
missing source matrix from the official BF16 checkpoint, silently requantize a preserved FP8 or
NVFP4 field, or add unused source calibration fields as artifact objects.

## 13. Existing `groupwise-int` artifact

The existing registered peer artifact retains this identity:

```text
filename   = qwen3_8_27b.ninfer
model_id   = qwen3.8-27b
weights_id = groupwise-int
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b-v1
```

It contains 1118 tensors and the same six resources. Its complete ordered inventory, logical row
views, aliases, and writer order are defined by `tools/convert/qwen3_8_27b/inventory.py`. Its format
counts are:

| Format | Tensors |
|---|---:|
| `BF16` | 582 |
| `FP32` | 96 |
| `I32` | 1 |
| `Q4G64_F16S` | 183 |
| `Q5G64_F16S` | 246 |
| `Q6G64_F16S` | 1 |
| `W8G32_F16S` | 9 |

`text/token_embedding [248320,5120]` and `text/output_head [248320,5120]` use `W8G32_F16S`; Text
layers use the registered Q4/Q5 allocation; the optimized draft head uses Q4; MTP matrices and
the Vision merger use W8; and the Vision patch projection uses Q6. All groupwise integer tensors
use `MAXABS_F16_RECIP_RNE_V1` with `row-split-k128-v1`. The artifact is produced solely from the
official source revision in Section 10.1 and binds through the `Qwen38GroupwiseInt` profile. Its
registered NVFP4 peer binds through `Qwen38Nvfp4`.

Its canonical conversion entry point remains:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.ninfer \
  --device cuda
```

The converter validates the official checkpoint, frontend resources, complete object plan, and
numeric recipes before opening the output, then writes the sibling
`qwen3_8_27b.ninfer.conversion.json` report.
