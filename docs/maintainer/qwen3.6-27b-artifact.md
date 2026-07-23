# Qwen3.6-27B Artifact and Conversion Reference

This reference records the complete `.ninfer` object inventory and conversion recipe for the
exact `qwen3.6-27b` checkpoint: object names, shapes, formats, layouts, frontend resources, source
transforms, logical views, and binder obligations. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics in
[`qwen3.6-27b-model.md`](qwen3.6-27b-model.md).

## 1. Artifact identity and boundary

The exact registered model identity is:

```text
qwen3.6-27b
```

The source/tool target key for this conversion implementation is
`qwen3_6_27b`. It selects target-private converter and reference code; it is not a second
`model_id` and is not serialized in the artifact JSON.

Every artifact with this `model_id` is one complete product route. It contains all Text, optimized
draft-head, MTP, Vision, and frontend objects listed here. There are no Text-only, no-Vision,
no-MTP, full-head-only, alternate-quantization, or optional-resource variants. File offsets and
physical object order are artifact-instance facts; the accepted object inventory and each object's
storage signature are model-contract facts.

The boundary is deliberate:

```text
.ninfer JSON
    model_id, persistent object name, kind, shape, format/layout or encoding,
    payload-relative offset, and byte length

this artifact contract
    exact required object set, storage signatures, logical views, aliases,
    frontend filenames, and binder invariants

conversion recipe in this document and target-private converter code
    Hugging Face source keys, slicing, reshaping, concatenation, derived values,
    quantization assignment, and draft-head construction
```

The conversion recipe is not serialized into the `.ninfer` JSON. In particular, the artifact does
not carry Hugging Face tensor names, source shards, layer/module enums, transforms, ranking-file
paths, fusion records, a model graph, execution phases, or kernel choices. Descriptive conversion
provenance belongs in the external conversion sidecar.

## 2. Fixed checkpoint facts used by the storage contract

All tensor shapes below use conventional logical row-major notation. A matrix `[N,K]` has `N`
output rows and `K` input columns; the quantized formats group along `K`.

| Fact | Value |
|---|---:|
| Text vocabulary rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| Text hidden width | 5120 |
| Text layers | 64 |
| Text MLP intermediate width | 17408 |
| full-attention layers | 16 |
| GDN layers | 48 |
| query heads / KV heads | 24 / 4 |
| attention head width | 256 |
| query rows / KV rows | 6144 / 1024 |
| GDN key heads × width | 16 × 128 = 2048 |
| GDN value heads × width | 48 × 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| MTP layers | 1 |
| draft-head rows | 131072 |
| Vision depth / hidden width | 27 / 1152 |
| Vision MLP width / heads | 4304 / 16 |
| Vision patch input width | 3 × 2 × 16 × 16 = 1536 |
| Vision position rows | 2304 |
| Vision merger input / output | 4608 / 5120 |

Full-attention Text layers are exactly:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every other layer in `0..63` is a GDN layer. Layer numbers in object names are unpadded decimal
integers.

## 3. Complete object namespace and physical order

### 3.1 Naming rules

The names in this section are exact artifact binding names. They are not Hugging Face source names.

- `text/` owns the token embedding, 64 Text layers, final norm, full output head, and draft head.
- `mtp/` owns the one-layer MTP-specific persistent weights. Its shared embedding and full head are
  logical aliases to `text/` objects and are not stored twice.
- `vision/` owns the complete Vision tower and merger.
- `frontend/` owns the six raw checkpoint resources consumed by the native C++ Frontend and the
  independent Python reference.
- Numeric-format spellings do not appear in object names. The JSON `format` member is authoritative.
- Names identify physical stored objects. Q/K, gate/value, gate/up, and other row views in Section 8
  are not additional JSON objects.

### 3.2 Canonical writer order

The target converter writes objects in this order:

1. the six frontend resources in Section 4 order;
2. `text/token_embedding`;
3. Text layers in ascending layer order, using the applicable table order in Sections 5.2 or 5.3;
4. `text/final_norm`, then `text/output_head`;
5. the two draft-head objects in Section 5.4 order;
6. the twelve MTP objects in Section 6 order;
7. the three Vision stem objects, 27 Vision blocks in ascending order, and six merger objects in
   Section 7 order.

Readers and binders select by name, not by array index. A different legal file placement does not
change a role, but the project converter uses the order above for predictable sequential writing and
inspection.

## 4. Frontend resources

The artifact contains exactly these six resource objects. All use `raw-bytes-v1` and preserve the
source file bytes without an internal archive or filename header.

| Order | Object name | Encoding | Source filename | Runtime meaning |
|---:|---|---|---|---|
| 0 | `frontend/tokenizer.json` | `raw-bytes-v1` | `tokenizer.json` | BPE vocabulary, merges, token bytes, and added-token ids through 248069 |
| 1 | `frontend/tokenizer_config.json` | `raw-bytes-v1` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `frontend/chat_template.jinja` | `raw-bytes-v1` | `chat_template.jinja` | registered Qwen template identity and semantics |
| 3 | `frontend/generation_config.json` | `raw-bytes-v1` | `generation_config.json` | model-default stop token IDs |
| 4 | `frontend/preprocessor_config.json` | `raw-bytes-v1` | `preprocessor_config.json` | image resize, normalization, and patch limits |
| 5 | `frontend/video_preprocessor_config.json` | `raw-bytes-v1` | `video_preprocessor_config.json` | video sampling, resize, normalization, and frame limits |

These six payloads are byte-identical to the corresponding 35B-A3B resources. Both artifacts
therefore carry the same pinned official Qwen3.6 family resource set:

| Object | SHA-256 |
|---|---|
| `frontend/tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `frontend/tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| `frontend/chat_template.jinja` | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| `frontend/generation_config.json` | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `frontend/preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `frontend/video_preprocessor_config.json` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |

`vocab.json`, `merges.txt`, `added_tokens.json`, and `special_tokens_map.json` are not objects in
this route. The official source does not use the latter two files. The native tokenizer reads the
base vocabulary, merges, and added-token subset from `tokenizer.json`, merges the agreeing
definitions from `tokenizer_config.json::added_tokens_decoder`, and appends its config-only ids
`248070..248076`. Conflicting ids, contents, flags, duplicate mappings, or a hole in the registered
`0..248076` domain are invalid.

The official processor resources omit class defaults rather than serializing expanded values. The
native processor resolves `rescale_factor=1/255`, video `fps=2`, `min_frames=4`, and
`max_frames=768`; any explicitly present value must agree. The compiled renderer follows the
official template: it accepts at most one leading `system` message, rejects direct `developer` and
late `system` roles, requires a real user query, and serializes non-string tool arguments as JSON.

The C++ binder retains these payloads as owned strings and constructs the shared Qwen3.6 native
tokenizer, template renderer, and image/video processor directly from them; it does not create a
temporary checkpoint directory. The independent Python reference may materialize the six source
filenames in a temporary directory and consume them through Transformers. Conversion validates the
exact official six-resource hash profile. The native route validates the official pad-token policy,
registered tokenizer domain, required Vision tokens, and processor configuration; the Python route
validates the tokenizer domain, Vision tokens, and required library processor inputs. MRoPE
positions and `rope_delta` are derived prepared-prompt values rather than resource contents.

## 5. Text and draft-head tensor inventory

Every quantized tensor in this section uses `row-split-k128-v1`. Every direct tensor uses
`contiguous-le-v1`.

### 5.1 Text-global objects

| Order | Object name | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `text/token_embedding` | `[248320,5120]` | `Q6G64_F16S` | `row-split-k128-v1` |
| after all layers | `text/final_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| after final norm | `text/output_head` | `[248320,5120]` | `Q6G64_F16S` | `row-split-k128-v1` |

`text/token_embedding` and `text/output_head` are distinct stored matrices. The checkpoint has
`tie_word_embeddings=false`. MTP reuses both roles logically; that reuse does not tie the two Text
objects to each other.

### 5.2 Full-attention Text layer

For every full-attention layer `l` in the exact list in Section 2, emit these nine objects in table
order:

| Order | Object-name pattern | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 1 | `text/layers/{l}/attention/query_key` | `[7168,5120]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 2 | `text/layers/{l}/attention/gate_value` | `[7168,5120]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 3 | `text/layers/{l}/attention/query_norm` | `[256]` | `BF16` | `contiguous-le-v1` |
| 4 | `text/layers/{l}/attention/key_norm` | `[256]` | `BF16` | `contiguous-le-v1` |
| 5 | `text/layers/{l}/attention/output` | `[5120,6144]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 6 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 7 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 8 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `Q5G64_F16S` | `row-split-k128-v1` |

### 5.3 GDN Text layer

For every other layer `l` in `0..63`, emit these thirteen objects in table order:

| Order | Object-name pattern | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 1 | `text/layers/{l}/gdn/a_log` | `[48]` | `FP32` | `contiguous-le-v1` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[48]` | `FP32` | `contiguous-le-v1` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,10240]` | `BF16` | `contiguous-le-v1` |
| 4 | `text/layers/{l}/gdn/a_projection` | `[48,5120]` | `BF16` | `contiguous-le-v1` |
| 5 | `text/layers/{l}/gdn/b_projection` | `[48,5120]` | `BF16` | `contiguous-le-v1` |
| 6 | `text/layers/{l}/gdn/query_key` | `[4096,5120]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 7 | `text/layers/{l}/gdn/value_z` | `[12288,5120]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 8 | `text/layers/{l}/gdn/norm` | `[128]` | `BF16` | `contiguous-le-v1` |
| 9 | `text/layers/{l}/gdn/output` | `[5120,6144]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 10 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 11 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 12 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `Q5G64_F16S` | `row-split-k128-v1` |

`gdn/value_z` owns two logical matrices: value is rows `[0,6144)` and z is rows
`[6144,12288)`. RowSplit planes cover the complete parent independently, so their physical order
is `base(value), base(z), high(value), high(z), scale(value), scale(z)`; concatenating two complete
encoded child payloads would be invalid.

The convolution shape is intentionally `[tap,channel] = [4,10240]`. Its bytes are tap-major, so
this is the stored shape. A channel-major `[10240,4]` consumer view is
`stored.transpose(0,1)`; it is not a plain contiguous reshape.

### 5.4 Optimized draft head

These two objects are mandatory and follow `text/output_head`:

| Order | Object name | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `text/draft_head` | `[131072,5120]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` | `contiguous-le-v1` |

Row `i` of `text/draft_head` represents full-head row
`text/draft_head_token_ids[i]`. The ID map contains 131072 unique values in `0..248076`; proposal
argmax returns a shortlist row and must be remapped before it becomes a token ID. Target prefill,
ordinary target decode, and MTP target verification always use `text/output_head`.

## 6. MTP tensor inventory

The one-layer MTP block contains exactly these twelve objects in table order:

| Order | Object name | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `mtp/input_projection` | `[5120,10240]` | `W8G32_F16S` | `row-split-k128-v1` |
| 1 | `mtp/embedding_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 2 | `mtp/hidden_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 3 | `mtp/layer/input_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 4 | `mtp/layer/attention/query_key_gate_value` | `[14336,5120]` | `W8G32_F16S` | `row-split-k128-v1` |
| 5 | `mtp/layer/attention/query_norm` | `[256]` | `BF16` | `contiguous-le-v1` |
| 6 | `mtp/layer/attention/key_norm` | `[256]` | `BF16` | `contiguous-le-v1` |
| 7 | `mtp/layer/attention/output` | `[5120,6144]` | `W8G32_F16S` | `row-split-k128-v1` |
| 8 | `mtp/layer/post_attention_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 9 | `mtp/layer/mlp/gate_up` | `[34816,5120]` | `W8G32_F16S` | `row-split-k128-v1` |
| 10 | `mtp/layer/mlp/down` | `[5120,17408]` | `W8G32_F16S` | `row-split-k128-v1` |
| 11 | `mtp/final_norm` | `[5120]` | `BF16` | `contiguous-le-v1` |

MTP has no duplicate embedding or full output head. Its logical `token_embedding` role aliases
`text/token_embedding`; its full target/proposal-head role aliases `text/output_head`; the selected
optimized proposal path additionally uses `text/draft_head` and `text/draft_head_token_ids`.

The artifact contains every persistent weight needed for shifted MTP prefill, autoregressive draft
proposal, full target verification, accepted-prefix continuation, correction/bonus production, and
multimodal MTP alignment. KV, GDN snapshots, positions, accepted counts, and proposal state are
runtime state and are not artifact objects.

## 7. Vision tensor inventory

Every quantized Vision tensor uses `row-split-k128-v1`; every direct tensor uses
`contiguous-le-v1`.

### 7.1 Vision stem

| Order | Object name | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `vision/patch_embedding` | `[1152,1536]` | `Q6G64_F16S` | `row-split-k128-v1` |
| 1 | `vision/patch_embedding_bias` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 2 | `vision/position_embedding` | `[2304,1152]` | `BF16` | `contiguous-le-v1` |

### 7.2 Vision transformer block

For every Vision block `b` in `0..26`, emit these twelve objects in table order:

| Order | Object-name pattern | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `vision/layers/{b}/attention/qkv` | `[3456,1152]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 1 | `vision/layers/{b}/attention/qkv_bias` | `[3456]` | `BF16` | `contiguous-le-v1` |
| 2 | `vision/layers/{b}/attention/output` | `[1152,1152]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 3 | `vision/layers/{b}/attention/output_bias` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 4 | `vision/layers/{b}/mlp/fc1` | `[4304,1152]` | `Q4G64_F16S` | `row-split-k128-v1` |
| 5 | `vision/layers/{b}/mlp/fc1_bias` | `[4304]` | `BF16` | `contiguous-le-v1` |
| 6 | `vision/layers/{b}/mlp/fc2` | `[1152,4304]` | `Q5G64_F16S` | `row-split-k128-v1` |
| 7 | `vision/layers/{b}/mlp/fc2_bias` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 8 | `vision/layers/{b}/norm1/weight` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 9 | `vision/layers/{b}/norm1/bias` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 10 | `vision/layers/{b}/norm2/weight` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 11 | `vision/layers/{b}/norm2/bias` | `[1152]` | `BF16` | `contiguous-le-v1` |

### 7.3 Vision merger

| Order | Object name | Shape | Format | Layout |
|---:|---|---|---|---|
| 0 | `vision/merger/fc1` | `[4608,4608]` | `W8G32_F16S` | `row-split-k128-v1` |
| 1 | `vision/merger/fc1_bias` | `[4608]` | `BF16` | `contiguous-le-v1` |
| 2 | `vision/merger/fc2` | `[5120,4608]` | `W8G32_F16S` | `row-split-k128-v1` |
| 3 | `vision/merger/fc2_bias` | `[5120]` | `BF16` | `contiguous-le-v1` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `BF16` | `contiguous-le-v1` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `BF16` | `contiguous-le-v1` |

These objects cover the complete patch projection, learned position table, 27 transformer blocks,
and merger. The artifact stores no resized pixels, video frames, patch grids, MRoPE positions,
visual embeddings, or Vision workspace; those are request data or transient execution state.

## 8. Logical views and aliases

The following half-open row ranges are part of the target contract. They are formed by the binder;
they are not serialized as objects. A view inherits its parent object's format, K dimension, and
row-split storage geometry.

| Parent object | Logical role | Row range | Logical shape |
|---|---|---|---|
| `text/layers/{l}/attention/query_key` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| `text/layers/{l}/attention/gate_value` | output gate | `[0,6144)` | `[6144,5120]` |
| same | value | `[6144,7168)` | `[1024,5120]` |
| `text/layers/{l}/gdn/query_key` | GDN query | `[0,2048)` | `[2048,5120]` |
| same | GDN key | `[2048,4096)` | `[2048,5120]` |
| `text/layers/{l}/gdn/value_z` | GDN value | `[0,6144)` | `[6144,5120]` |
| same | GDN z | `[6144,12288)` | `[6144,5120]` |
| `text/layers/{l}/mlp/gate_up` | MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MLP up | `[17408,34816)` | `[17408,5120]` |
| `mtp/layer/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `mtp/layer/mlp/gate_up` | MTP MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MTP MLP up | `[17408,34816)` | `[17408,5120]` |

The full-attention views exist only for the 16 full-attention layer numbers. The GDN query/key and
value/z views exist only for the other 48 layer numbers. All 64 Text layers have the MLP gate/up
views. The table defines 16 row-view templates and produces 390 bound row views for this target.

Additional fixed aliases are:

| Logical consumer role | Stored object(s) |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution taps | transpose of `text/layers/{l}/gdn/convolution` from `[4,10240]` to `[10240,4]` |

The old 1314 segment records and 130 fusion records therefore disappear from persistent metadata.
The named physical objects and this fixed view table contain all information the registered target
needs.

## 9. Inventory proof

### 9.1 Component counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `16 × 9` | 144 |
| GDN Text layers | `48 × 13` | 624 |
| Text total | `3 + 144 + 624` | 771 |
| optimized draft head | weight + ID map | 2 |
| MTP | fixed table | 12 |
| Vision stem | fixed table | 3 |
| Vision transformer blocks | `27 × 12` | 324 |
| Vision merger | fixed table | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| all tensors | `771 + 2 + 12 + 333` | 1118 |
| frontend resources | fixed table | 6 |
| complete artifact | `1118 + 6` | 1124 |

### 9.2 Numeric-format counts

| Format | Text | draft head | MTP | Vision | Total |
|---|---:|---:|---:|---:|---:|
| `BF16` | 353 | 0 | 7 | 222 | 582 |
| `FP32` | 96 | 0 | 0 | 0 | 96 |
| `I32` | 0 | 1 | 0 | 0 | 1 |
| `Q4G64_F16S` | 128 | 1 | 0 | 54 | 183 |
| `Q5G64_F16S` | 192 | 0 | 0 | 54 | 246 |
| `Q6G64_F16S` | 2 | 0 | 0 | 1 | 3 |
| `W8G32_F16S` | 0 | 0 | 5 | 2 | 7 |
| total | 771 | 2 | 12 | 333 | 1118 |

The three direct formats account for 679 `contiguous-le-v1` tensors. The four quantized formats
account for 439 `row-split-k128-v1` tensors.

## 10. Conversion recipe boundary

Sections 10 through 15 define how the selected source checkpoint produces the artifact contract
above. They govern the target-private converter and its external conversion report. They do not add
members to the artifact JSON and are not needed by a runtime binder.

The selected source is the official `Qwen/Qwen3.6-27B` resource/configuration set pinned at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Its fifteen retained BF16 shards have the same LFS
SHA-256 object ids as that revision. Conversion report recipe id
`qwen3_6_27b-v2` denotes this official source-resource contract; tensor recipes, formats,
layouts, counts, and byte totals are unchanged from v1.

The source checkout path is not part of `model_id` or an artifact validity condition. The recipe
identifies source tensors by the patterns below, and the converter resolves
them through `model.safetensors.index.json` one object at a time.

## 11. Source checkpoint preflight

Before writing payloads, the converter checks the source configuration and source inventory. The
current selected checkpoint has the following required facts:

| Configuration path | Required value |
|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` |
| `model_type` | `"qwen3_5"` |
| `language_model_only` | `false` |
| `text_config.hidden_size` | 5120 |
| `text_config.num_hidden_layers` | 64 |
| `text_config.intermediate_size` | 17408 |
| `text_config.vocab_size` | 248320 |
| `text_config.num_attention_heads` / `num_key_value_heads` | 24 / 4 |
| `text_config.head_dim` | 256 |
| `text_config.full_attention_interval` | 4 |
| `text_config.linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 |
| `text_config.linear_num_value_heads` / `linear_value_head_dim` | 48 / 128 |
| `text_config.linear_conv_kernel_dim` | 4 |
| `text_config.mamba_ssm_dtype` | `"float32"` |
| `text_config.mtp_num_hidden_layers` | 1 |
| `text_config.mtp_use_dedicated_embeddings` | `false` |
| `text_config.tie_word_embeddings` | `false` |
| `text_config.max_position_embeddings` | 262144 |
| `text_config.rms_norm_eps` | `1e-6` |
| `text_config.rope_parameters.rope_theta` | 10000000 |
| `text_config.rope_parameters.mrope_section` | `[11,11,10]` |
| `vision_config.depth` / `hidden_size` | 27 / 1152 |
| `vision_config.intermediate_size` / `num_heads` | 4304 / 16 |
| `vision_config.in_channels` | 3 |
| `vision_config.temporal_patch_size` / `patch_size` | 2 / 16 |
| `vision_config.spatial_merge_size` | 2 |
| `vision_config.num_position_embeddings` | 2304 |
| `vision_config.out_hidden_size` | 5120 |
| root `tie_word_embeddings` | `false` |
| root Vision token IDs | start 248053, end 248054, image 248056, video 248057 |

`text_config.layer_types` must contain exactly 64 entries and must agree with the full/GDN layer
set in Section 2. Every source tensor used by Sections 12 through 14 must exist, have the stated
shape, and use BF16 source words. The current recipe references 1199 distinct source tensors; all
1199 exist in the selected checkpoint and are BF16.

Before source tensors are opened for conversion, all six resource payloads must match the official
SHA-256 profile in Section 4. MTP depth is read only from
`text_config.mtp_num_hidden_layers`; a root-level alias is neither required nor accepted as source
authority.

## 12. Common numeric conversion rules

### 12.1 Direct objects

- A BF16 artifact object copies the source BF16 logical words after any explicitly stated reshape
  or transpose.
- `gdn/a_log` and `gdn/dt_bias` expand their source BF16 values exactly to FP32 and store the
  resulting IEEE binary32 words.
- The draft-head token map is constructed as integers and range-checked before conversion to I32;
  it is not converted from a floating source tensor.

No other implicit dtype conversion is part of this recipe.

### 12.2 Quantized objects

Every quantized object uses the canonical symmetric per-row encoder profile
`MAXABS_F16_RECIP_RNE_V1` from
[`tensor-formats.md`](tensor-formats.md), with the selected object's registered format:

1. place the intended input axis last and obtain a logical `[N,K]` BF16 matrix;
2. expand source values to FP32 for scale and code calculation;
3. append zero columns to `K_pad = align_up(K,128)`;
4. quantize every row and every G64 or G32 group independently;
5. calculate `max_abs/qmax`, round the stored scale to binary16, and form the binary32 reciprocal in
   the profile's specified arithmetic order;
6. if a nonzero group's scale underflows binary16 to zero, use the smallest positive binary16
   subnormal `2^-24`;
7. multiply each value by that reciprocal, round codes to nearest with ties-to-even, then clamp to
   the format's legal interval;
8. encode codes and scales with `row-split-k128-v1`.

Zero K-padding participates only as zero values and cannot increase a logical group's `max_abs`.
The artifact records the unpadded logical shape from Sections 5 through 7; the layout derives
`K_pad` and the exact payload size.

## 13. Text and draft-head source mapping

Let:

```text
P(l) = model.language_model.layers.{l}.
```

### 13.1 Global Text sources

| Artifact object | Hugging Face source | Transform |
|---|---|---|
| `text/token_embedding` | `model.language_model.embed_tokens.weight` `[248320,5120]` | quantize Q6 |
| `text/final_norm` | `model.language_model.norm.weight` `[5120]` | preserve BF16 |
| `text/output_head` | `lm_head.weight` `[248320,5120]` | quantize Q6 |

### 13.2 Full-attention layer sources

The source query projection has shape `[12288,5120]`. Its rows are interleaved per head as
`[query(256), gate(256)]` for each of 24 heads. Define:

```text
q_proj = source.reshape(24, 512, 5120)
query  = q_proj[:, 0:256, :].reshape(6144, 5120)
gate   = q_proj[:, 256:512, :].reshape(6144, 5120)
```

A simple first-half/second-half split is incorrect.

| Artifact object suffix under `text/layers/{l}/` | Hugging Face source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight` `[5120]` | preserve BF16 |
| `attention/query_key` | `self_attn.q_proj.weight`, `self_attn.k_proj.weight` `[1024,5120]` | extract `query`, then row-concatenate `[query,k]`, quantize Q4 |
| `attention/gate_value` | same q-proj, `self_attn.v_proj.weight` `[1024,5120]` | extract `gate`, then row-concatenate `[gate,v]`, quantize Q5 |
| `attention/query_norm` | `self_attn.q_norm.weight` `[256]` | preserve BF16 |
| `attention/key_norm` | `self_attn.k_norm.weight` `[256]` | preserve BF16 |
| `attention/output` | `self_attn.o_proj.weight` `[5120,6144]` | quantize Q5 |
| `post_attention_norm` | `post_attention_layernorm.weight` `[5120]` | preserve BF16 |
| `mlp/gate_up` | `mlp.gate_proj.weight`, `mlp.up_proj.weight`, each `[17408,5120]` | row-concatenate `[gate,up]`, quantize Q4 |
| `mlp/down` | `mlp.down_proj.weight` `[5120,17408]` | quantize Q5 |

### 13.3 GDN layer sources

The fused source `linear_attn.in_proj_qkv.weight` has shape `[10240,5120]` and row order:

```text
query = [0,2048)
key   = [2048,4096)
value = [4096,10240)
```

| Artifact object suffix under `text/layers/{l}/` | Hugging Face source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight` `[5120]` | preserve BF16 |
| `gdn/a_log` | `linear_attn.A_log` `[48]` BF16 | expand to FP32 |
| `gdn/dt_bias` | `linear_attn.dt_bias` `[48]` BF16 | expand to FP32 |
| `gdn/convolution` | `linear_attn.conv1d.weight` `[10240,1,4]` | take `source[:,0,:]`, transpose to contiguous `[4,10240]` |
| `gdn/a_projection` | `linear_attn.in_proj_a.weight` `[48,5120]` | preserve BF16 |
| `gdn/b_projection` | `linear_attn.in_proj_b.weight` `[48,5120]` | preserve BF16 |
| `gdn/query_key` | `linear_attn.in_proj_qkv.weight` | row-concatenate query then key, equivalently rows `[0,4096)`, quantize Q4 |
| `gdn/value_z` | `linear_attn.in_proj_qkv.weight`, `linear_attn.in_proj_z.weight` `[6144,5120]` | concatenate QKV rows `[4096,10240)` then z as BF16 `[value,z]`, quantize the `[12288,5120]` parent once as Q5 |
| `gdn/norm` | `linear_attn.norm.weight` `[128]` | preserve BF16 |
| `gdn/output` | `linear_attn.out_proj.weight` `[5120,6144]` | quantize Q5 |
| `post_attention_norm` | `post_attention_layernorm.weight` `[5120]` | preserve BF16 |
| `mlp/gate_up` | `mlp.gate_proj.weight`, `mlp.up_proj.weight`, each `[17408,5120]` | row-concatenate `[gate,up]`, quantize Q4 |
| `mlp/down` | `mlp.down_proj.weight` `[5120,17408]` | quantize Q5 |

### 13.4 Draft-head construction

The draft head is derived before quantization from `lm_head.weight` and the project ranking file:

```text
tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
```

The ranking file is a little-endian I64 array reshaped to `[rows,248320]`; only row 0, the total
frequency count, participates. The recipe is:

1. read every tokenizer entry in `tokenizer_config.json` whose
   `added_tokens_decoder[id].special` is true and form the sorted forced-ID set;
2. stable-sort all vocabulary IDs by descending row-0 count; because the initial IDs are ascending,
   non-forced count ties use ascending token ID;
3. select the first `131072 - len(forced)` non-forced IDs;
4. append the sorted forced IDs and stable-sort the selected sequence by descending count;
5. require exactly 131072 unique IDs, each in `0..248076`;
6. write those IDs as `text/draft_head_token_ids` I32 in resulting order;
7. gather the same rows from the BF16 `lm_head.weight` and quantize them as
   `text/draft_head` Q4G64.

The ranking path, counts, source tokenizer path, and selection procedure are recipe/provenance facts,
not `.ninfer` metadata. The ID map is the only persistent information needed to interpret the
shortlist rows.

## 14. MTP and Vision source mapping

### 14.1 MTP sources

Let `M = mtp.layers.0.`. The MTP q-projection uses the same per-head
`[query(256),gate(256)]` row interleave as the Text q-projection and is split by the same reshape
defined in Section 13.2.

| Artifact object | Hugging Face source | Transform |
|---|---|---|
| `mtp/input_projection` | `mtp.fc.weight` `[5120,10240]` | quantize W8G32 |
| `mtp/embedding_norm` | `mtp.pre_fc_norm_embedding.weight` `[5120]` | preserve BF16 |
| `mtp/hidden_norm` | `mtp.pre_fc_norm_hidden.weight` `[5120]` | preserve BF16 |
| `mtp/layer/input_norm` | `M + input_layernorm.weight` `[5120]` | preserve BF16 |
| `mtp/layer/attention/query_key_gate_value` | `M + self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight` | extract and concatenate `[query,k,gate,v]`, quantize W8G32 |
| `mtp/layer/attention/query_norm` | `M + self_attn.q_norm.weight` `[256]` | preserve BF16 |
| `mtp/layer/attention/key_norm` | `M + self_attn.k_norm.weight` `[256]` | preserve BF16 |
| `mtp/layer/attention/output` | `M + self_attn.o_proj.weight` `[5120,6144]` | quantize W8G32 |
| `mtp/layer/post_attention_norm` | `M + post_attention_layernorm.weight` `[5120]` | preserve BF16 |
| `mtp/layer/mlp/gate_up` | `M + mlp.gate_proj.weight`, `up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize W8G32 |
| `mtp/layer/mlp/down` | `M + mlp.down_proj.weight` `[5120,17408]` | quantize W8G32 |
| `mtp/final_norm` | `mtp.norm.weight` `[5120]` | preserve BF16 |

### 14.2 Vision sources

All Vision source keys begin with `model.visual.`. Except for the patch projection reshape, each
stored object reads the same semantic source suffix named in the table.

| Artifact object | Hugging Face source suffix | Transform |
|---|---|---|
| `vision/patch_embedding` | `patch_embed.proj.weight` `[1152,3,2,16,16]` | contiguous reshape to `[1152,1536]`, quantize Q6 |
| `vision/patch_embedding_bias` | `patch_embed.proj.bias` `[1152]` | preserve BF16 |
| `vision/position_embedding` | `pos_embed.weight` `[2304,1152]` | preserve BF16 |

For every Vision block `b`, the source prefix is `model.visual.blocks.{b}.`:

| Artifact object suffix under `vision/layers/{b}/` | Source suffix | Transform |
|---|---|---|
| `attention/qkv` | `attn.qkv.weight` `[3456,1152]` | quantize Q4 |
| `attention/qkv_bias` | `attn.qkv.bias` `[3456]` | preserve BF16 |
| `attention/output` | `attn.proj.weight` `[1152,1152]` | quantize Q5 |
| `attention/output_bias` | `attn.proj.bias` `[1152]` | preserve BF16 |
| `mlp/fc1` | `mlp.linear_fc1.weight` `[4304,1152]` | quantize Q4 |
| `mlp/fc1_bias` | `mlp.linear_fc1.bias` `[4304]` | preserve BF16 |
| `mlp/fc2` | `mlp.linear_fc2.weight` `[1152,4304]` | quantize Q5 |
| `mlp/fc2_bias` | `mlp.linear_fc2.bias` `[1152]` | preserve BF16 |
| `norm1/weight` | `norm1.weight` `[1152]` | preserve BF16 |
| `norm1/bias` | `norm1.bias` `[1152]` | preserve BF16 |
| `norm2/weight` | `norm2.weight` `[1152]` | preserve BF16 |
| `norm2/bias` | `norm2.bias` `[1152]` | preserve BF16 |

The merger source prefix is `model.visual.merger.`:

| Artifact object suffix under `vision/merger/` | Source suffix | Transform |
|---|---|---|
| `fc1` | `linear_fc1.weight` `[4608,4608]` | quantize W8G32 |
| `fc1_bias` | `linear_fc1.bias` `[4608]` | preserve BF16 |
| `fc2` | `linear_fc2.weight` `[5120,4608]` | quantize W8G32 |
| `fc2_bias` | `linear_fc2.bias` `[5120]` | preserve BF16 |
| `norm/weight` | `norm.weight` `[1152]` | preserve BF16 |
| `norm/bias` | `norm.bias` `[1152]` | preserve BF16 |

## 15. Binder obligations

After the generic reader has established the common container and layout invariants, the
Qwen3.6-27B binder must perform exactly the model-specific work below.

1. Require `model_id == "qwen3.6-27b"`.
2. Generate the complete name/signature inventory from Sections 4 through 7 and require exactly
   those 1124 objects: no missing object, duplicate role, alternate signature, or extra profile
   object.
3. Require every tensor's exact shape, format, and layout and every resource's exact
   `raw-bytes-v1` encoding. Encoded byte sizes come from the registered layouts, not this binder.
4. Always materialize immutable Text bindings for the 16 full-attention layers, 48 GDN layers, all
   64 MLPs, embedding, final norm, and full head. Validate the draft head and ID map in every
   artifact, but materialize them only for the startup-selected optimized MTP proposal path.
5. Build the fixed row views and aliases in Section 8. These views are derived once during binding;
   Text/MTP/Vision hot paths do not perform artifact-name lookups.
6. Treat each `[4,10240]` GDN convolution as tap-major and form the consumer's channel-major
   transpose explicitly.
7. Require 131072 unique draft-head IDs in `0..248076` and bind row `i` to its corresponding mapped
   vocabulary ID.
8. Validate every MTP tensor. Materialize and bind MTP to the shared Text embedding and full head
   only when the startup draft window is nonzero, while keeping MTP KV/state separate at execution
   time.
9. Validate the complete 27-layer Vision and merger inventory. Materialize it only when Vision is
   enabled at startup; a request never changes that decision.
10. Retain the six frontend resource payloads for the loaded-model lifetime and construct the
    shared Qwen3.6 native Frontend from them. Verify the family resource identity, tokenizer domain,
    required special IDs, template identity, and processor configuration before generation.
11. Reject an incompletely validated product. Publish Text plus exactly the MTP, optimized proposal,
    and Vision groups selected at startup only after complete artifact validation succeeds.

The binder does not know source checkpoint paths, safetensors keys, ranking inputs, quantization
operations, converter arguments, or conversion sidecar fields. Conversely, the common `.ninfer`
reader does not know any name, shape, layer count, view, alias, token ID, or completeness rule in
this document.

## 16. Implementation locations and checks

The target converter in `tools/convert/qwen3_6_27b/` and the independent binder in
`tools/reference/qwen3_6_27b/` derive this inventory and agree through a real emitted
artifact. The verifier and reference path have established:

- 1118 tensor objects and six frontend resources;
- component counts `771 + 2 + 12 + 333`;
- numeric counts `582 BF16`, `96 FP32`, `1 I32`, `183 Q4`, `246 Q5`, `3 Q6`, and `7 W8`;
- layout counts `679 contiguous-le-v1` and `439 row-split-k128-v1`;
- 48 physical `gdn/value_z` parents and 96 logical value/z row views, with no physical
  `gdn/value` or `gdn/z` object;
- all 1199 referenced BF16 source tensors present with expected shapes;
- the six resources consumable by both the native C++ Frontend and the selected Hugging Face
  library path used by the Python reference;
- complete Text, image/video Vision, MTP, and combined multimodal reference execution from the
  resulting `.ninfer` artifact.

The 2026-07-17 atomic cutover compared every unaffected descriptor and payload byte exactly and
compared all retired value/z code, high-bit, and scale planes with their new parent row ranges
before replacing the canonical artifact. The canonical artifact was then re-inspected, re-verified
against the source checkpoint, and loaded through the public C++ Engine. The recipe ID and common
container version did not change, and no compatibility binder or obsolete artifact was retained.

The 2026-07-18 official-checkpoint cutover preserved every shard inode, size, mtime, and content id,
then regenerated and atomically promoted the canonical artifact with recipe id
`qwen3_6_27b-v2`. Its 1118-tensor descriptor signature and component byte totals are
unchanged; its six resource payloads now match Section 4. The promoted bytes passed source
verification, artifact-native Transformers frontend checks, and a registered C++ Engine greedy
smoke. The existing 35B-A3B artifact already matched the same resource profile and was retained.

These are structural, numerical, and behavioral checks of the selected route. They do not require
a fixed file hash, byte-identical regeneration, or exact reproduction of a probabilistic token
stream.
