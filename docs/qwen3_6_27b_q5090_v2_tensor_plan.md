# Qwen3.6-27B q5090 v2 Tensor Plan & Block/Fusion Assignment

This document is the **authoritative per-tensor assignment** for the v2 packed weight format. It is the
input contract the converter (`tools/q5090_convert/tensor_plan.py`) implements and the unit test
asserts.

The same block/segment/fusion assignment applies to v3 unchanged; v3 changes only the on-disk
code-plane packing. The full v3 artifact includes TEXT_CORE, MTP_DRAFT, and VISION_ENCODER:
1167 blocks, 1311 segments, 128 fusion groups.

It composes two layers:

- **Quantization policy** — *which qtype each tensor gets* — is owned by
  [qwen3_6_27b_q5090_final_quant_format_v1.md](qwen3_6_27b_q5090_final_quant_format_v1.md) and is
  **carried over unchanged** (v2 is value-preserving; see that doc for the qtype rationale).
- **v2 storage assignment** — *which block, segment, and fusion group each tensor lands in* — is **new
  and owned here**, on top of the byte contract in
  [q5090_packed_file_format_v2.md](q5090_packed_file_format_v2.md).

This is not a migration of v1: there is no v1 tiled layout. Every quantized tensor is `ROW_SPLIT`;
every dense control tensor is `CONTIGUOUS` (§7 of the binary spec).

## 0. Terminology

- **Block** = one `TensorEntry` = one `ROW_SPLIT` (or `CONTIGUOUS`) matrix, single `qtype`, single `K`,
  `N` rows, code plane + scale plane.
- **Segment** = one logical projection = a contiguous row-range of a block, named and identified by
  `source_kind` + `source_layer` + canonical name.
- **Row / segment semantics by layout.** For `ROW_SPLIT` blocks, `N` is the output-row count and the
  segments partition `[0, N)`. For `CONTIGUOUS` control blocks (rank-1 `[5120]`/`[256]`/`[48]`, rank-3
  `conv1d [10240,4,1]`) there is exactly one segment with `row_begin = 0`, `row_count = shape[0]`
  (leading dim) and `scale_plane_bytes = 0`; the row-partition and code/scale-plane notions are
  `ROW_SPLIT`-only (binary spec §5).
- **Fusion group** = the set of blocks that consume the same input activation at the same schedule
  point, bound by a `FusionGroupRecord`.

Block payloads for the members of a fusion group are emitted consecutively (binary spec §6).

## 1. Model geometry (locked)

| field | value | derived |
|---|---:|---|
| `hidden` | 5120 | linear input dim |
| `intermediate` | 17408 | MLP gate/up out, down in |
| `vocab` | 248320 | embed / lm_head rows |
| `n_layers` | 64 | layers 0..63 |
| `full_attention_interval` | 4 | full-attn where `(layer+1) % 4 == 0` |
| full-attn layers | 3,7,11,…,63 | 16 layers |
| GDN (linear_attention) layers | the other 48 | 48 layers |
| attn heads / kv heads / head_dim | 24 / 4 / 256 | q/gate = 24·256 = 6144; k/v = 4·256 = 1024 |
| GDN key/value heads, head dims | 16·128 / 48·128 | key_dim = 2048; value_dim = 6144; conv_dim = 2·2048+6144 = 10240 |

## 2. qtype assignment (carried from policy, unchanged values)

| projection class | qtype |
|---|---|
| attn `q`, `k`; GDN `in_q`, `in_k`; MLP `gate`, `up` | `Q4G64_F16S` |
| attn `gate`, `v`, `o_proj`; GDN `in_v`, `in_z`, `out_proj`; MLP `down` | `Q5G64_F16S` |
| `embed_tokens`, `lm_head` | `Q6G64_F16S` |
| all norms, GDN `A_log`/`dt_bias`/`conv1d`/`in_a`/`in_b`/`norm` | `BF16_CTRL` / `FP32_CTRL` |
| MTP linears; vision merger FC | `W8G128_F16S` |
| vision block linears, patch_embed | `Q4G64_F16S` / `Q5G64_F16S` |

`group_size = 64` for Q4/Q5/Q6, `128` for W8. Dequant values are bit-identical to the policy quantizer
(binary spec invariant 0.1).

## 3. Conversion transforms (carried from v1, applied before quantization)

These produce the logical projection tensors that are then quantized and laid out. They are
value-defining (correctness-critical) and unchanged by v2:

- **`attn q_proj` de-interleave.** HF `self_attn.q_proj.weight` `[12288,5120]` packs `[query|gate]`
  interleaved per head (`view[24, 2·256, 5120]`). The converter extracts `query → ATTN_Q [6144,5120]`
  and `gate → ATTN_GATE [6144,5120]`. A naive contiguous `[0:6144]/[6144:]` split scrambles heads.
- **GDN `in_proj_qkv` row split.** `[10240,5120]` → `in_q = rows[0:2048]`, `in_k = rows[2048:4096]`,
  `in_v = rows[4096:10240]` (contiguous in the reference).
- **GDN `conv1d` runtime-native.** `[C,1,K]` → metadata shape `[C,K,1]` with flat bytes in `[K,C]`
  order (`conv_dim=10240, K=4`).

## 4. TEXT_CORE assignment

### 4.1 Full-attention layer template (16 instances: L ∈ {3,7,…,63})

Quantized blocks (`ROW_SPLIT`):

| block (canonical name) | qtype | `[N,K]` | fusion (id, idx) | segments: `source_kind` `[row_begin,row_end)` |
|---|---|---:|---|---|
| `layers.{L}.attn_in.q4` | Q4 | `[7168,5120]` | `ATTN_IN`, 0 | `ATTN_Q [0,6144)`, `ATTN_K [6144,7168)` |
| `layers.{L}.attn_in.q5` | Q5 | `[7168,5120]` | `ATTN_IN`, 1 | `ATTN_GATE [0,6144)`, `ATTN_V [6144,7168)` |
| `layers.{L}.self_attn.o_proj.weight` | Q5 | `[5120,6144]` | NONE | `ATTN_O [0,5120)` |
| `layers.{L}.mlp.gateup` | Q4 | `[34816,5120]` | `MLP_GATEUP`, 0 | `MLP_GATE [0,17408)`, `MLP_UP [17408,34816)` |
| `layers.{L}.mlp.down_proj.weight` | Q5 | `[5120,17408]` | NONE | `MLP_DOWN [0,5120)` |

Control blocks (`CONTIGUOUS`):

| block | qtype | shape | `source_kind` |
|---|---|---:|---|
| `layers.{L}.input_layernorm.weight` | BF16 | `[5120]` | `INPUT_LAYERNORM` |
| `layers.{L}.self_attn.q_norm.weight` | BF16 | `[256]` | `ATTN_Q_NORM` |
| `layers.{L}.self_attn.k_norm.weight` | BF16 | `[256]` | `ATTN_K_NORM` |
| `layers.{L}.post_attention_layernorm.weight` | BF16 | `[5120]` | `POST_ATTN_LAYERNORM` |

Per full layer: **9 blocks, 12 segments, 2 fusion groups** (`ATTN_IN`, `MLP_GATEUP`).

### 4.2 GDN (linear_attention) layer template (48 instances)

Quantized blocks (`ROW_SPLIT`):

| block (canonical name) | qtype | `[N,K]` | fusion (id, idx) | segments: `source_kind` `[row_begin,row_end)` |
|---|---|---:|---|---|
| `layers.{L}.gdn_in.q4` | Q4 | `[4096,5120]` | `GDN_IN`, 0 | `GDN_IN_PROJ_Q [0,2048)`, `GDN_IN_PROJ_K [2048,4096)` |
| `layers.{L}.gdn_in.q5` | Q5 | `[6144,5120]` | `GDN_IN`, 1 | `GDN_IN_PROJ_V [0,6144)` |
| `layers.{L}.linear_attn.in_proj_z.weight` | Q5 | `[6144,5120]` | NONE | `GDN_IN_PROJ_Z [0,6144)` |
| `layers.{L}.linear_attn.out_proj.weight` | Q5 | `[5120,6144]` | NONE | `GDN_OUT_PROJ [0,5120)` |
| `layers.{L}.mlp.gateup` | Q4 | `[34816,5120]` | `MLP_GATEUP`, 0 | `MLP_GATE [0,17408)`, `MLP_UP [17408,34816)` |
| `layers.{L}.mlp.down_proj.weight` | Q5 | `[5120,17408]` | NONE | `MLP_DOWN [0,5120)` |

`in_z` is **standalone**, not in `GDN_IN`: `z` is computed late (after the conv/gating/recurrence) and
feeds only the output-gate norm, so fusing it would force early computation and extend a `[6144,T]`
activation across the recurrence in prefill, for no decode benefit (binary spec §10).

Control blocks (`CONTIGUOUS`):

| block | qtype | shape | `source_kind` |
|---|---|---:|---|
| `layers.{L}.input_layernorm.weight` | BF16 | `[5120]` | `INPUT_LAYERNORM` |
| `layers.{L}.linear_attn.A_log` | FP32 | `[48]` | `GDN_A_LOG` |
| `layers.{L}.linear_attn.dt_bias` | FP32 | `[48]` | `GDN_DT_BIAS` |
| `layers.{L}.linear_attn.conv1d.weight` | BF16 | `[10240,4,1]` | `GDN_CONV1D` |
| `layers.{L}.linear_attn.in_proj_a.weight` | BF16 | `[48,5120]` | `GDN_IN_PROJ_A` |
| `layers.{L}.linear_attn.in_proj_b.weight` | BF16 | `[48,5120]` | `GDN_IN_PROJ_B` |
| `layers.{L}.linear_attn.norm.weight` | BF16 | `[128]` | `GDN_NORM` |
| `layers.{L}.post_attention_layernorm.weight` | BF16 | `[5120]` | `POST_ATTN_LAYERNORM` |

Per GDN layer: **14 blocks, 16 segments, 2 fusion groups** (`GDN_IN`, `MLP_GATEUP`).

> Control tensors are stored `CONTIGUOUS` and do not participate in `ROW_SPLIT` planes or fusion (each
> is a single-segment block, §0). Their shapes come from HF **except `conv1d`**, which applies the
> canonical runtime-native transform of §3 (`[C,1,K]` → metadata `[C,K,1]`, flat bytes `[K,C]`); that
> transform is value-defining and is not overridden by "from HF". `A_log`/`dt_bias` are FP32
> (exponent-sensitive); all other control is BF16.

### 4.3 Globals

| block (canonical name) | qtype | layout | `[N,K]` / shape | fusion | segment |
|---|---|---|---:|---|---|
| `model.language_model.embed_tokens.weight` | Q6 | `ROW_SPLIT` | `[248320,5120]` | NONE | `EMBED [0,248320)` |
| `model.language_model.norm.weight` | BF16 | `CONTIGUOUS` | `[5120]` | NONE | `FINAL_NORM` |
| `lm_head.weight` | Q6 | `ROW_SPLIT` | `[248320,5120]` | NONE | `LM_HEAD [0,248320)` |

`embed_tokens` is gather-driven; it uses `ROW_SPLIT` for layout uniformity (binary spec §13 note).

### 4.4 TEXT_CORE counts and consistency

| | full ×16 | GDN ×48 | globals | total |
|---|---:|---:|---:|---:|
| blocks | 144 | 672 | 3 | **819** |
| segments | 192 | 768 | 3 | **963** |
| fusion groups | 32 | 96 | 0 | **128** |

Consistency checks (the converter and a v2 reader must satisfy both):

- **`segment_count` (963) equals the v1 TEXT_CORE `tensor_count` (963).** Segments are exactly the
  logical projections; fusion changes how they are *stored*, not how many there are.
- **`block_count` (819) = 963 − 144 fused merges.** The 144 merges are: `ATTN_IN.q4` (16),
  `ATTN_IN.q5` (16), `GDN_IN.q4` (48), `MLP_GATEUP` (64), each merging 2 projections into 1 block.

### 4.5 Canonical emission / index order

The converter walks TEXT_CORE in this order; tensor index order == payload order. Fusion-group members
are emitted consecutively to satisfy the `FusionGroupRecord` adjacency invariant.

```text
embed_tokens
for L in 0..63:
  input_layernorm
  if full(L):
    attn_in.q4, attn_in.q5            # ATTN_IN group (consecutive)
    self_attn.q_norm, self_attn.k_norm
    self_attn.o_proj
  else: # GDN
    linear_attn.A_log, dt_bias, conv1d, in_proj_a, in_proj_b
    gdn_in.q4, gdn_in.q5              # GDN_IN group (consecutive)
    linear_attn.norm
    linear_attn.in_proj_z
    linear_attn.out_proj
  post_attention_layernorm
  mlp.gateup                          # MLP_GATEUP group (single block)
  mlp.down_proj
model.language_model.norm             # final norm
lm_head
```

### 4.6 Block / segment naming

- **Standalone block** `name`/`name_hash` = the projection's canonical HF-derived name (the names in
  the tables above); its single segment carries the same name.
- **Fused block** `name`/`name_hash` = the converter-assigned group name (`…attn_in.q4`,
  `…gdn_in.q5`, `…mlp.gateup`); each segment carries its own projection canonical name (e.g.
  `…self_attn.q_proj.q`, `…mlp.gate_proj.weight`). `TensorEntry.source_kind` of a fused block is
  `OTHER` (0); segment identity is authoritative (binary spec §4 rule).

## 5. MTP_DRAFT (included; standalone assignment rules)

MTP is included in the full artifact; its detailed fusion plan is deferred. v2/v3 layout rules for
the 15 MTP tensors (one MTP layer + globals):

- W8 linears (`self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`, `mtp.fc`) → `ROW_SPLIT`,
  `group_size=128`, **standalone** (MTP fusion groups deferred).
- norms (`input_layernorm`, `q_norm`, `k_norm`, `post_attention_layernorm`,
  `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`) → `CONTIGUOUS` BF16.

Exact per-tensor list and shapes: see `tools/q5090_convert/tensor_plan.py::build_mtp_specs` and the
policy doc. MTP `source_layer` is its own index space.

## 6. VISION_ENCODER (included; standalone assignment rules)

Vision is included in the full artifact as `LAZY_GPU` and is not on the text decode path; fusion is
deferred. v2/v3 layout rules
(depth = 27 blocks, plus patch_embed / pos_embed / merger):

- block linears `attn.qkv` (Q4), `attn.proj` (Q5), `mlp.linear_fc1` (Q4), `mlp.linear_fc2` (Q5),
  `patch_embed.proj` (Q5, reshaped `[1152,1536]`) → `ROW_SPLIT` `group_size=64`, **standalone**.
- `merger.linear_fc1/fc2` (W8) → `ROW_SPLIT` `group_size=128`, **standalone**.
- all biases, `norm1/norm2/merger.norm`, `pos_embed` → `CONTIGUOUS` BF16.
- vision `intermediate_size=4304` needs K-padding to a multiple of 64 (4352) for `fc1`/`fc2`; all text
  dims are already aligned.

Exact per-tensor list: `tools/q5090_convert/tensor_plan.py::build_vision_specs` and the policy doc.

## 7. Open / deferred decisions

- **MTP and vision fusion groups** are intentionally deferred (standalone for now). When MTP is
  introduced, `MTP` attention/MLP inputs may gain `ATTN_IN`/`MLP_GATEUP`-style fusion under new
  `fusion_group_id` values; this document and the binary-spec enum will be extended then.
- **`embed`/`lm_head` are untied** (two separate Q6 blocks). If a future model ties them, the ABI
  expresses it as **one** Q6 block with **one** segment, bound by **two consumers** at the runtime
  level (the gather and the lm_head GEMV both reference that block) — *not* as two overlapping segments
  spanning the same rows (segments must partition `[0, N)`, so two identities cannot both cover the
  full row range).
