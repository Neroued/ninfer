# Qwen3.6-35B-A3B Artifact Reference

This reference defines the complete `.ninfer` object inventory for `qwen3.6-35b-a3b`: object
names, order, shapes, numeric formats, storage layouts, fused row order, aliases, frontend
resources, and source-to-object transforms. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics in
[`qwen3.6-35b-a3b-model.md`](qwen3.6-35b-a3b-model.md).

## 1. Artifact identity and contents

The registered model identity is:

```text
qwen3.6-35b-a3b
```

The source/tool/compiled target key is:

```text
qwen3_6_35b_a3b
```

The target key selects this exact checkpoint package and is not serialized as a second `model_id`.

Every conforming artifact is one complete product image containing Text, the optimized proposal
head, MTP, Vision, DFlash, and the six frontend resources in this document. DFlash is part of the
same artifact and does not define a second artifact or an optional artifact profile.

The first 889 objects retain the existing sequence. The 51 DFlash tensor objects are appended
after the Vision merger objects. Existing object names, order, shapes, formats, layouts,
payload-relative offsets, and stored values are unchanged.

The artifact JSON does not add target-private fields for expert count, fused roles, source tensor
names, layer classes, DFlash feature-layer ids, DFlash attention classes, DFlash block size, or the
DFlash mask id. These are fixed by this target contract.

## 2. Fixed target facts

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. Quantization groups
along `K`.

| Fact | Value |
|---|---:|
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| DFlash mask-token row | 248077 |
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
| optimized draft-head rows | 131072 |
| DFlash layers / hidden / intermediate width | 6 / 2048 / 6144 |
| DFlash query / KV heads / head width | 32 / 8 / 128 |
| DFlash target-feature layers | `[1,6,11,16,22,27,32,37]` |
| DFlash target-feature input width | `8 × 2048 = 16384` |
| DFlash block size / proposal positions | 16 / 15 |
| DFlash sliding-window layers / window | `0..4` / 4096 |
| DFlash full-context layer | 5 |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 |
| Vision heads / patch input | 16 / 1536 |
| Vision merger input / output | 4608 / 2048 |
| native context capacity | 262144 |

Full-attention Text layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39
```

Every other Text layer is GDN. Every Text and MTP decoder layer has 256 routed experts, selects
eight, and additionally runs one gated shared expert.

The frontend token domain is `0..248076`. `text/token_embedding` and `text/output_head` retain
248320 matrix rows. DFlash uses embedding row 248077 as its mask embedding; that row is not a
frontend token id.

The base `tokenizer.json` carries added-token ids through 248069, and
`tokenizer_config.json::added_tokens_decoder` supplies ids `248070..248076`. The frontend merges
both resources into the 248077-entry token domain.

## 3. Numeric formats and physical row order

### 3.1 Format assignment

| Role | Format |
|---|---|
| token embedding | `W8G32_F16S` |
| full output head | `Q6G64_F16S` |
| main full-attention and GDN large matrices | `W8G32_F16S` |
| main shared-expert gate/up/down | `W8G32_F16S` |
| routed gate/up, all 40 Text layers | `Q4G64_F16S` |
| routed down, Text layers except 34/38/39 | `Q5G64_F16S` |
| routed down, Text layers 34/38/39 | `Q6G64_F16S` |
| router and shared-expert scalar gate | `BF16` |
| Text norms, convolution, and A/B projections | `BF16` |
| GDN `A_log` and `dt_bias` | `FP32` |
| optimized proposal head | `Q4G64_F16S` |
| MTP matrices, including routed experts | `W8G32_F16S` |
| DFlash feature, attention, and MLP matrices | `W8G32_F16S` |
| DFlash norms | `BF16` |
| Vision block input/expansion matrices | `Q4G64_F16S` |
| Vision block output/contraction matrices | `Q5G64_F16S` |
| Vision patch projection | `Q6G64_F16S` |
| Vision merger matrices | `W8G32_F16S` |
| all other Vision weights and biases | `BF16` |

Every quantized tensor uses encoder profile `MAXABS_F16_RECIP_RNE_V1` and storage layout
`row-split-k128-v1`. Every direct tensor uses `contiguous-le-v1`.

### 3.2 Fused row order

The following concatenations define physical output-row order:

- full-attention input: `[query,key,output_gate,value]`;
- GDN input: `[query,key,value,z]`;
- GDN A/B projection: `[a_0,...,a_31,b_0,...,b_31]`;
- MoE router/shared gate: router rows `0..255`, then shared-expert gate row 256;
- Text and MTP dense gate/up: `[gate,up]`;
- every DFlash attention input: `[query,key,value]`;
- every DFlash MLP input: `[gate,up]`.

For a source routed gate/up bank `G[E,2I,K]` with `E=256` and `I=512`, the artifact stores
`[E*I*2,K] = [262144,2048]` with:

```text
stored_row(e, p, r) = e * 1024 + p * 512 + r
p = 0: gate
p = 1: up
```

For a source routed down bank `D[E,H,I]`, the artifact stores
`[E*H,I] = [524288,512]` with:

```text
stored_row(e, h) = e * 2048 + h
```

Expert ids are not permuted.

## 4. Object namespace, order, and frontend resources

### 4.1 Namespace and writer order

- `text/` contains the embedding, 40 Text layers, final norm, full output head, and optimized
  proposal head.
- `mtp/` contains MTP-private tensors.
- `vision/` contains the Vision tower and merger.
- `dflash/` contains DFlash-private tensors.
- `frontend/` contains the six raw frontend resources.

Objects are written in this order:

1. the six frontend resources in Section 4.2;
2. `text/token_embedding`;
3. Text layers `0..39`, using the applicable object order in Sections 5.2 and 5.3;
4. `text/final_norm`, `text/output_head`, `text/draft_head`, and
   `text/draft_head_token_ids`;
5. the fifteen MTP tensors in Section 6;
6. the Vision stem, blocks `0..26`, and merger in Section 7;
7. `dflash/feature_projection`, `dflash/context_norm`, DFlash layers `0..5`, and
   `dflash/final_norm`, using the order in Section 8.

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

## 5. Text and optimized proposal inventory

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,2048]` | `W8G32_F16S` |
| after all layers | `text/final_norm` | `[2048]` | `BF16` |
| next | `text/output_head` | `[248320,2048]` | `Q6G64_F16S` |

The embedding and output head are independent objects.

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

The convolution is stored tap-major as `[4,8192]`.

### 5.4 Routed-down precision set

The routed-down format is `Q6G64_F16S` for layers:

```text
34, 38, 39
```

It is `Q5G64_F16S` for the other 37 layers.

### 5.5 Optimized proposal head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,2048]` | `Q4G64_F16S` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` |

Row `i` of `text/draft_head` represents full-head row
`text/draft_head_token_ids[i]`. The ids are unique and lie in `0..248076`.

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

MTP token embedding, full output head, and optimized proposal head are aliases listed in
Section 9.3.

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
| 2 | `vision/merger/fc2` | `[2048,4608]` | `W8G32_F16S` |
| 3 | `vision/merger/fc2_bias` | `[2048]` | `BF16` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `BF16` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `BF16` |

No Vision deep-stack object exists.

## 8. DFlash inventory

### 8.1 DFlash-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `dflash/feature_projection` | `[2048,16384]` | `W8G32_F16S` |
| 1 | `dflash/context_norm` | `[2048]` | `BF16` |

### 8.2 DFlash layer

For every DFlash layer `l` in `0..5`, emit these eight objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `dflash/layers/{l}/input_norm` | `[2048]` | `BF16` |
| 1 | `dflash/layers/{l}/attention/query_key_value` | `[6144,2048]` | `W8G32_F16S` |
| 2 | `dflash/layers/{l}/attention/query_norm` | `[128]` | `BF16` |
| 3 | `dflash/layers/{l}/attention/key_norm` | `[128]` | `BF16` |
| 4 | `dflash/layers/{l}/attention/output` | `[2048,4096]` | `W8G32_F16S` |
| 5 | `dflash/layers/{l}/post_attention_norm` | `[2048]` | `BF16` |
| 6 | `dflash/layers/{l}/mlp/gate_up` | `[12288,2048]` | `W8G32_F16S` |
| 7 | `dflash/layers/{l}/mlp/down` | `[2048,6144]` | `W8G32_F16S` |

Layers `0..4` are sliding-window layers. Layer 5 is the full-context layer. Attention class is not
encoded in an object name or metadata field.

### 8.3 DFlash-final object

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `dflash/final_norm` | `[2048]` | `BF16` |

DFlash has no private token-embedding object, mask-embedding object, output-head object, or
shortlist-head object.

## 9. Logical views and aliases

All entries in this section are views or aliases of the physical objects above. They are not extra
artifact objects.

### 9.1 Dense fused views

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
| `dflash/layers/{l}/attention/query_key_value` | query | `[0,4096)` | `[4096,2048]` |
| same | key | `[4096,5120)` | `[1024,2048]` |
| same | value | `[5120,6144)` | `[1024,2048]` |
| `dflash/layers/{l}/mlp/gate_up` | gate | `[0,6144)` | `[6144,2048]` |
| same | up | `[6144,12288)` | `[6144,2048]` |

The `*` attention pattern applies to Text full-attention layers and the MTP layer.

For `gdn/a_b_projection`, rows `[0,32)` are A and rows `[32,64)` are B. For
`moe/router_shared_gate`, rows `[0,256)` are router rows and row 256 is the shared-expert gate.

For every Text or MTP shared gate/up object, row `r` is gate and row `512+r` is up.

### 9.2 Routed-expert views

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

### 9.3 Aliases

| Logical consumer role | Stored object or view |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| optimized proposal head for MTP or DFlash | `text/draft_head` plus `text/draft_head_token_ids` |
| DFlash anchor-token embedding | row view of `text/token_embedding` |
| DFlash mask embedding | row 248077 of `text/token_embedding` |
| DFlash full proposal output head | `text/output_head` |
| DFlash optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose view of the stored `[4,8192]` convolution |

The optimized proposal pair is independent of the selected drafter. It may be consumed by MTP or
DFlash, but is materialized only when the startup proposal-head route is `optimized`.

## 10. Inventory summary

### 10.1 Object counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `10 × 11` | 110 |
| GDN Text layers | `30 × 14` | 420 |
| main Text excluding draft | `3 + 110 + 420` | 533 |
| optimized proposal head | weight + id map | 2 |
| MTP | fixed inventory | 15 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 × 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| DFlash globals | feature projection + context norm | 2 |
| DFlash layers | `6 × 8` | 48 |
| DFlash final norm | fixed inventory | 1 |
| DFlash total | `2 + 48 + 1` | 51 |
| all tensors | `533 + 2 + 15 + 333 + 51` | 934 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `934 + 6` | 940 |

### 10.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | DFlash | Total |
|---|---:|---:|---:|---:|---:|---:|
| `BF16` | 231 | 0 | 8 | 222 | 26 | 487 |
| `FP32` | 60 | 0 | 0 | 0 | 0 | 60 |
| `I32` | 0 | 1 | 0 | 0 | 0 | 1 |
| `Q4G64_F16S` | 40 | 1 | 0 | 54 | 0 | 95 |
| `Q5G64_F16S` | 37 | 0 | 0 | 54 | 0 | 91 |
| `Q6G64_F16S` | 4 | 0 | 0 | 1 | 0 | 5 |
| `W8G32_F16S` | 161 | 0 | 7 | 2 | 25 | 195 |
| total | 533 | 2 | 15 | 333 | 51 | 934 |

The artifact contains 548 direct tensors using `contiguous-le-v1` and 386 quantized tensors using
`row-split-k128-v1`.

## 11. Source identities and numeric conversion

### 11.1 Source identities

| Component | Source identity | Tensor source |
|---|---|---|
| Text, optimized proposal head, MTP, Vision, and frontend | `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0` | `model.safetensors.index.json` and its 26 shards |
| DFlash | `z-lab/Qwen3.6-35B-A3B-DFlash` revision `f181eece646affea2c38b2765f1aaa01a9734ccd` | `dflash-bf16/model.safetensors` |

The base checkpoint contributes 1045 BF16 source tensors. The DFlash checkpoint contributes 69
BF16 source tensors. Sections 12 and 13 account for every checkpoint tensor and every artifact
tensor.

The six frontend files in Section 4.2 come from the base checkpoint and are retained under their
exact source filenames.

### 11.2 Direct tensors

- `BF16` objects preserve source BF16 words after the specified concatenate, reshape, or transpose.
- GDN `a_log` and `dt_bias` expand source BF16 values to `FP32`.
- `text/draft_head_token_ids` is stored as `I32`.

### 11.3 Quantized tensors

For every quantized artifact object:

1. complete the specified concatenate or reshape as a contiguous logical `[N,K]` BF16 matrix;
2. apply `MAXABS_F16_RECIP_RNE_V1` independently to every output row and every G32 or G64 group
   along `K`;
3. encode the resulting codes and binary16 scales with `row-split-k128-v1`.

Groups do not cross output rows. Layout padding does not change the logical shape recorded in the
artifact descriptor.

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

The source q-projection is `[8192,2048]` with per-head `[query_256,gate_256]` rows:

```text
qg    = q_proj.reshape(16,512,2048)
query = qg[:,0:256,:].reshape(4096,2048)
gate  = qg[:,256:512,:].reshape(4096,2048)
qkgv  = concatenate_rows(query, k_proj, gate, v_proj)
```

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight` | preserve BF16 |
| `attention/query_key_gate_value` | `self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight` | form `[q,k,gate,v]`, quantize W8 |
| `attention/query_norm` | `self_attn.q_norm.weight` | preserve BF16 |
| `attention/key_norm` | `self_attn.k_norm.weight` | preserve BF16 |
| `attention/output` | `self_attn.o_proj.weight [2048,4096]` | quantize W8 |
| `post_attention_norm` | `post_attention_layernorm.weight` | preserve BF16 |

### 12.3 GDN source transform

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

### 12.4 MoE transform shared by every Text layer

| Artifact suffix under `text/layers/{l}/` | Source under `P(l)` | Transform |
|---|---|---|
| `moe/router_shared_gate` | `mlp.gate.weight [256,2048]`, `mlp.shared_expert_gate.weight [1,2048]` | concatenate, preserve BF16 |
| `moe/routed_gate_up` | `mlp.experts.gate_up_proj [256,1024,2048]` | expert-major flatten, quantize Q4 |
| `moe/routed_down` | `mlp.experts.down_proj [256,2048,512]` | expert-major flatten, quantize Q5/Q6 by Section 5.4 |
| `moe/shared_gate_up` | shared `gate_proj.weight`, `up_proj.weight`, each `[512,2048]` | concatenate `[gate,up]`, quantize W8 |
| `moe/shared_down` | shared `down_proj.weight [2048,512]` | quantize W8 |

### 12.5 Optimized proposal-head construction

The fixed ranking source is:

```text
tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
```

Padded rows `248077..248319` are not candidates. Select ids from `0..248076` as follows:

1. form the ascending forced-id set from entries whose merged
   `added_tokens_decoder[id].special` value is true;
2. stable-sort all candidate ids by descending row-0 count, with ascending-id count ties;
3. take the first `131072 - len(forced)` non-forced ids;
4. append the ascending forced ids and stable-sort the result by descending count;
5. require exactly 131072 unique ids in `0..248076`;
6. store those ids as `I32` in `text/draft_head_token_ids`;
7. gather the same ordered BF16 rows from the 35B `lm_head.weight` and quantize them to
   `text/draft_head`.

## 13. MTP, Vision, and DFlash source mapping

### 13.1 MTP

Let `M = mtp.layers.0.`. Apply the Text full-attention q/gate extraction and MoE row transforms.

| Artifact object | Source | Transform |
|---|---|---|
| `mtp/input_projection` | `mtp.fc.weight [2048,4096]` | quantize W8 |
| `mtp/embedding_norm` | `mtp.pre_fc_norm_embedding.weight` | preserve BF16 |
| `mtp/hidden_norm` | `mtp.pre_fc_norm_hidden.weight` | preserve BF16 |
| `mtp/layer/input_norm` | `M + input_layernorm.weight` | preserve BF16 |
| `mtp/layer/attention/query_key_gate_value` | `M + self_attn.{q,k,v}_proj.weight` | form `[q,k,gate,v]`, quantize W8 |
| `mtp/layer/attention/query_norm` | `M + self_attn.q_norm.weight` | preserve BF16 |
| `mtp/layer/attention/key_norm` | `M + self_attn.k_norm.weight` | preserve BF16 |
| `mtp/layer/attention/output` | `M + self_attn.o_proj.weight` | quantize W8 |
| `mtp/layer/post_attention_norm` | `M + post_attention_layernorm.weight` | preserve BF16 |
| `mtp/layer/moe/router_shared_gate` | `M + mlp.gate.weight`, `M + mlp.shared_expert_gate.weight` | concatenate, preserve BF16 |
| `mtp/layer/moe/routed_gate_up` | `M + mlp.experts.gate_up_proj` | expert-major flatten, quantize W8 |
| `mtp/layer/moe/routed_down` | `M + mlp.experts.down_proj` | expert-major flatten, quantize W8 |
| `mtp/layer/moe/shared_gate_up` | `M + mlp.shared_expert.gate_proj.weight`, `M + mlp.shared_expert.up_proj.weight` | concatenate `[gate,up]`, quantize W8 |
| `mtp/layer/moe/shared_down` | `M + mlp.shared_expert.down_proj.weight` | quantize W8 |
| `mtp/final_norm` | `mtp.norm.weight` | preserve BF16 |

### 13.2 Vision

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
| `fc2` | `linear_fc2.weight [2048,4608]` | quantize W8 |
| `fc2_bias` | `linear_fc2.bias [2048]` | preserve BF16 |
| `norm/weight` | `norm.weight [1152]` | preserve BF16 |
| `norm/bias` | `norm.bias [1152]` | preserve BF16 |

### 13.3 DFlash

DFlash source tensor names are relative to `dflash-bf16/model.safetensors`.

| Artifact object | Source | Transform |
|---|---|---|
| `dflash/feature_projection` | `fc.weight [2048,16384]` | quantize W8 |
| `dflash/context_norm` | `hidden_norm.weight [2048]` | preserve BF16 |
| `dflash/final_norm` | `norm.weight [2048]` | preserve BF16 |

For every DFlash layer `l` in `0..5`:

| Artifact suffix under `dflash/layers/{l}/` | Source under `layers.{l}.` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [2048]` | preserve BF16 |
| `attention/query_key_value` | `self_attn.q_proj.weight [4096,2048]`, `k_proj.weight [1024,2048]`, `v_proj.weight [1024,2048]` | concatenate `[q,k,v]`, quantize W8 |
| `attention/query_norm` | `self_attn.q_norm.weight [128]` | preserve BF16 |
| `attention/key_norm` | `self_attn.k_norm.weight [128]` | preserve BF16 |
| `attention/output` | `self_attn.o_proj.weight [2048,4096]` | quantize W8 |
| `post_attention_norm` | `post_attention_layernorm.weight [2048]` | preserve BF16 |
| `mlp/gate_up` | `mlp.gate_proj.weight [6144,2048]`, `mlp.up_proj.weight [6144,2048]` | concatenate `[gate,up]`, quantize W8 |
| `mlp/down` | `mlp.down_proj.weight [2048,6144]` | quantize W8 |
