# Qwen3.6-27B Artifact Reference

This reference defines the complete `.ninfer` object inventory for `qwen3.6-27b`: object names,
order, shapes, numeric formats, storage layouts, fused row order, aliases, frontend resources, and
source-to-object transforms. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics in
[`qwen3.6-27b-model.md`](qwen3.6-27b-model.md).

## 1. Artifact identity and contents

The registered model identity is:

```text
qwen3.6-27b
```

The source/tool/compiled target key is:

```text
qwen3_6_27b
```

The target key selects this exact checkpoint package and is not serialized as a second `model_id`.

Every conforming artifact is one complete product image containing Text, the optimized MTP draft
head, MTP, Vision, and the six frontend resources in this document. These components do not define
separate artifacts or optional artifact profiles.

The artifact contains exactly 1118 tensor objects and six resource objects. Logical row views and
aliases are not additional objects or JSON records.

## 2. Fixed target facts

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. Quantization groups
along `K`.

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
| GDN key heads × width | 16 × 128 = 2048 |
| GDN value heads × width | 48 × 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| MTP layers | 1 full-attention dense-MLP layer |
| optimized draft-head rows | 131072 |
| Vision depth / hidden / intermediate width | 27 / 1152 / 4304 |
| Vision heads / patch input width | 16 / 1536 |
| Vision position rows | 2304 |
| Vision merger input / output | 4608 / 5120 |
| native context capacity | 262144 |

Full-attention Text layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every other Text layer is GDN. Layer numbers in object names are unpadded decimal integers.

The frontend token domain is `0..248076`. Both vocabulary matrices retain 248320 rows. The base
`tokenizer.json` carries added-token ids through 248069, and
`tokenizer_config.json::added_tokens_decoder` supplies ids `248070..248076`.

## 3. Numeric formats and physical row order

### 3.1 Format assignment

| Role | Format |
|---|---|
| token embedding | `Q6G64_F16S` |
| full output head | `Q6G64_F16S` |
| full-attention query/key | `Q4G64_F16S` |
| full-attention gate/value and output | `Q5G64_F16S` |
| GDN query/key | `Q4G64_F16S` |
| GDN value/z and output | `Q5G64_F16S` |
| Text MLP gate/up | `Q4G64_F16S` |
| Text MLP down | `Q5G64_F16S` |
| Text norms, GDN convolution, and GDN A/B projections | `BF16` |
| GDN `A_log` and `dt_bias` | `FP32` |
| optimized MTP draft head | `Q4G64_F16S` |
| optimized MTP draft-head id map | `I32` |
| MTP matrices | `W8G32_F16S` |
| MTP norms | `BF16` |
| Vision block input/expansion matrices | `Q4G64_F16S` |
| Vision block output/contraction matrices | `Q5G64_F16S` |
| Vision patch projection | `Q6G64_F16S` |
| Vision merger matrices | `W8G32_F16S` |
| all other Vision weights and biases | `BF16` |

Every quantized tensor uses encoder profile `MAXABS_F16_RECIP_RNE_V1` and storage layout
`row-split-k128-v1`. Every direct tensor uses `contiguous-le-v1`.

### 3.2 Fused row order

The following concatenations define physical output-row order:

- full-attention `query_key`: `[query,key]`;
- full-attention `gate_value`: `[output_gate,value]`;
- GDN `query_key`: `[query,key]`;
- GDN `value_z`: `[value,z]`;
- Text and MTP MLP input: `[gate,up]`;
- MTP attention input: `[query,key,output_gate,value]`.

Within every source full-attention q-projection, each of the 24 heads stores
`[query_256,output_gate_256]`. The converter separates those per-head halves before constructing
the artifact parents.

`gdn/value_z` is one quantized parent. Its code and scale planes cover the complete
`[12288,5120]` parent in `[value,z]` row order.

Every GDN convolution is stored tap-major as `[4,10240]`.

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
3. Text layers `0..63`, using the applicable object order in Sections 5.2 and 5.3;
4. `text/final_norm` and `text/output_head`;
5. `text/draft_head` and `text/draft_head_token_ids`;
6. the twelve MTP tensors in Section 6;
7. the Vision stem, blocks `0..26`, and merger in Section 7.

Readers bind by name. Object names contain no format spelling.

### 4.2 Frontend resources

The artifact contains exactly these `raw-bytes-v1` resources:

| Order | Object name | Source filename | Meaning |
|---:|---|---|---|
| 0 | `frontend/tokenizer.json` | `tokenizer.json` | base BPE vocabulary, merges, token bytes, and its added-token subset |
| 1 | `frontend/tokenizer_config.json` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `frontend/chat_template.jinja` | `chat_template.jinja` | registered Qwen template |
| 3 | `frontend/generation_config.json` | `generation_config.json` | default stop ids |
| 4 | `frontend/preprocessor_config.json` | `preprocessor_config.json` | image preprocessing limits and constants |
| 5 | `frontend/video_preprocessor_config.json` | `video_preprocessor_config.json` | video sampling and preprocessing |

The artifact retains these resource payloads byte-for-byte.

## 5. Text and optimized MTP draft inventory

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,5120]` | `Q6G64_F16S` |
| after all layers | `text/final_norm` | `[5120]` | `BF16` |
| next | `text/output_head` | `[248320,5120]` | `Q6G64_F16S` |

The embedding and output head are independent objects.

### 5.2 Full-attention Text layer

For every full-attention layer `l` in Section 2, emit these nine objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/attention/query_key` | `[7168,5120]` | `Q4G64_F16S` |
| 2 | `text/layers/{l}/attention/gate_value` | `[7168,5120]` | `Q5G64_F16S` |
| 3 | `text/layers/{l}/attention/query_norm` | `[256]` | `BF16` |
| 4 | `text/layers/{l}/attention/key_norm` | `[256]` | `BF16` |
| 5 | `text/layers/{l}/attention/output` | `[5120,6144]` | `Q5G64_F16S` |
| 6 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |
| 7 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `Q4G64_F16S` |
| 8 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `Q5G64_F16S` |

### 5.3 GDN Text layer

For every other layer `l` in `0..63`, emit these thirteen objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/gdn/a_log` | `[48]` | `FP32` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[48]` | `FP32` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,10240]` | `BF16` |
| 4 | `text/layers/{l}/gdn/a_projection` | `[48,5120]` | `BF16` |
| 5 | `text/layers/{l}/gdn/b_projection` | `[48,5120]` | `BF16` |
| 6 | `text/layers/{l}/gdn/query_key` | `[4096,5120]` | `Q4G64_F16S` |
| 7 | `text/layers/{l}/gdn/value_z` | `[12288,5120]` | `Q5G64_F16S` |
| 8 | `text/layers/{l}/gdn/norm` | `[128]` | `BF16` |
| 9 | `text/layers/{l}/gdn/output` | `[5120,6144]` | `Q5G64_F16S` |
| 10 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |
| 11 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `Q4G64_F16S` |
| 12 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `Q5G64_F16S` |

### 5.4 Optimized MTP draft head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,5120]` | `Q4G64_F16S` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` |

Row `i` of `text/draft_head` represents full-head row
`text/draft_head_token_ids[i]`. The ids are unique and lie in `0..248076`.

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

MTP token embedding, full output head, and optimized proposal head are aliases listed in
Section 8.2.

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

All entries in this section are views or aliases of the physical objects above. They are not extra
artifact objects.

### 8.1 Fused row views

| Parent object | Logical role | Stored row selection | Shape |
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

### 8.2 Aliases

| Logical consumer role | Stored object or view |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose view of the stored `[4,10240]` convolution |

## 9. Inventory summary

### 9.1 Object counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `16 × 9` | 144 |
| GDN Text layers | `48 × 13` | 624 |
| main Text excluding draft | `3 + 144 + 624` | 771 |
| optimized MTP draft head | weight + id map | 2 |
| MTP | fixed inventory | 12 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 × 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| all tensors | `771 + 2 + 12 + 333` | 1118 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `1118 + 6` | 1124 |

### 9.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | Total |
|---|---:|---:|---:|---:|---:|
| `BF16` | 353 | 0 | 7 | 222 | 582 |
| `FP32` | 96 | 0 | 0 | 0 | 96 |
| `I32` | 0 | 1 | 0 | 0 | 1 |
| `Q4G64_F16S` | 128 | 1 | 0 | 54 | 183 |
| `Q5G64_F16S` | 192 | 0 | 0 | 54 | 246 |
| `Q6G64_F16S` | 2 | 0 | 0 | 1 | 3 |
| `W8G32_F16S` | 0 | 0 | 5 | 2 | 7 |
| total | 771 | 2 | 12 | 333 | 1118 |

The artifact contains 679 direct tensors using `contiguous-le-v1` and 439 quantized tensors using
`row-split-k128-v1`.

## 10. Source identity and numeric conversion

### 10.1 Source identity

Text, optimized MTP draft-head, MTP, Vision, and frontend content come from
`Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Tensor sources are resolved through `model.safetensors.index.json` and its fifteen BF16 shards.

The checkpoint contributes 1199 BF16 source tensors. Sections 11 and 12 account for every
checkpoint tensor and every artifact tensor. The six frontend files in Section 4.2 are retained
under their exact source filenames.

### 10.2 Direct tensors

- `BF16` objects preserve source BF16 words after the specified concatenate, reshape, or transpose.
- GDN `a_log` and `dt_bias` expand source BF16 values to `FP32`.
- `text/draft_head_token_ids` is stored as `I32`.

### 10.3 Quantized tensors

For every quantized artifact object:

1. complete the specified split, concatenate, or reshape as a contiguous logical `[N,K]` BF16
   matrix;
2. apply `MAXABS_F16_RECIP_RNE_V1` independently to every output row and every G32 or G64 group
   along `K`;
3. encode the resulting codes and binary16 scales with `row-split-k128-v1`.

Groups do not cross output rows. Layout padding does not change the logical shape recorded in the
artifact descriptor.

## 11. Text source mapping

Let:

```text
P(l) = model.language_model.layers.{l}.
```

### 11.1 Text globals

| Artifact object | Source | Transform |
|---|---|---|
| `text/token_embedding` | `model.language_model.embed_tokens.weight [248320,5120]` | quantize Q6 |
| `text/final_norm` | `model.language_model.norm.weight [5120]` | preserve BF16 |
| `text/output_head` | `lm_head.weight [248320,5120]` | quantize Q6 |

### 11.2 Full-attention source transform

The source q-projection is `[12288,5120]` with per-head `[query_256,output_gate_256]` rows:

```text
qg    = q_proj.reshape(24,512,5120)
query = qg[:,0:256,:].reshape(6144,5120)
gate  = qg[:,256:512,:].reshape(6144,5120)
```

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` | preserve BF16 |
| `attention/query_key` | `self_attn.q_proj.weight`, `k_proj.weight [1024,5120]` | form `[query,key]`, quantize Q4 |
| `attention/gate_value` | `self_attn.q_proj.weight`, `v_proj.weight [1024,5120]` | form `[output_gate,value]`, quantize Q5 |
| `attention/query_norm` | `self_attn.q_norm.weight [256]` | preserve BF16 |
| `attention/key_norm` | `self_attn.k_norm.weight [256]` | preserve BF16 |
| `attention/output` | `self_attn.o_proj.weight [5120,6144]` | quantize Q5 |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` | preserve BF16 |
| `mlp/gate_up` | `mlp.gate_proj.weight`, `up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize Q4 |
| `mlp/down` | `mlp.down_proj.weight [5120,17408]` | quantize Q5 |

### 11.3 GDN source transform

The source `linear_attn.in_proj_qkv.weight [10240,5120]` row order is:

```text
query = [0,2048)
key   = [2048,4096)
value = [4096,10240)
```

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` | preserve BF16 |
| `gdn/a_log` | `linear_attn.A_log [48]` | expand to FP32 |
| `gdn/dt_bias` | `linear_attn.dt_bias [48]` | expand to FP32 |
| `gdn/convolution` | `linear_attn.conv1d.weight [10240,1,4]` | take `[:,0,:]`, transpose to `[4,10240]` |
| `gdn/a_projection` | `linear_attn.in_proj_a.weight [48,5120]` | preserve BF16 |
| `gdn/b_projection` | `linear_attn.in_proj_b.weight [48,5120]` | preserve BF16 |
| `gdn/query_key` | `linear_attn.in_proj_qkv.weight` | take rows `[0,4096)`, quantize Q4 |
| `gdn/value_z` | `linear_attn.in_proj_qkv.weight`, `linear_attn.in_proj_z.weight [6144,5120]` | concatenate `[value,z]`, quantize Q5 |
| `gdn/norm` | `linear_attn.norm.weight [128]` | preserve BF16 |
| `gdn/output` | `linear_attn.out_proj.weight [5120,6144]` | quantize Q5 |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` | preserve BF16 |
| `mlp/gate_up` | `mlp.gate_proj.weight`, `up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize Q4 |
| `mlp/down` | `mlp.down_proj.weight [5120,17408]` | quantize Q5 |

### 11.4 Optimized MTP draft-head construction

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
7. gather the same ordered BF16 rows from `lm_head.weight` and quantize them to
   `text/draft_head`.

## 12. MTP and Vision source mapping

### 12.1 MTP

Let `M = mtp.layers.0.`. Apply the Text full-attention q/gate extraction.

| Artifact object | Source | Transform |
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

### 12.2 Vision

All sources begin with `model.visual.`.

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
