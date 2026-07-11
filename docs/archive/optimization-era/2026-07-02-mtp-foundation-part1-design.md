# MTP 支持预先准备工作第一部分：q5090 MTP Layout 与 Python Ref Model

本文定义支持 Qwen3.6-27B MTP head 前，底层需要先补齐的第一部分工作：

- q5090 v3 权重格式侧的 MTP head layout 改造；
- Python 量化/打包脚本需要如何生成新的 MTP_DRAFT block/segment；
- Python ref model 需要如何加载并执行 MTP，用作 layout 修改的收尾验证。

本文不是引擎接入计划，不涉及 speculative decode 调度、Engine API、C++ model card 或
acceptance/rejection sampler。本文也不引入新的文件版本号；所有约定都在现有
`q5090_w4g64_mixed_v3` 容器内原地收敛。MTP_DRAFT 是精度敏感且体积较小的 draft model，本
阶段在 v3 内新增 `W8G32_F16S` qtype，并只将 MTP_DRAFT dense/fused linears 切到该 qtype；
TEXT_CORE 不含 W8，VISION_ENCODER 既有 `W8G128_F16S` merger FC 不属于本阶段修改范围。

## 1. Scope

### 1.1 In Scope

1. 将 MTP_DRAFT 的 linears 从当前 standalone 形式改为更接近 runtime-native 的 fused block
   布局：

   ```text
   mtp.fc
   mtp.attn_in.w8
   mtp.mlp.gateup.w8
   mtp.o_proj
   mtp.mlp.down
   ```

2. 明确这些 block 的 q5090 `module_kind`、`source_kind`、`source_layer`、segment row range、
   qtype、shape、source transform。
3. 明确 Python converter/verify 侧需要改哪些概念和 manifest 生成逻辑。
4. 明确 Python ref model 如何读取新 fused MTP layout，并实现 MTP 单步推理路径。
5. 明确 ref model 侧 MTP 状态边界、shared embedding/lm_head 复用方式和验收标准。

### 1.2 Out Of Scope

1. 不修改 Engine、runtime scheduler、CUDA graph、KV cache allocation/ownership、GDN/MTP
   runtime state lifecycle、proposal/verification state machine 或 target hidden-state handoff。
   Python ref model 可以维护自己的 correctness-first MTP KV 字典，但这不定义 C++ runtime
   的状态生命周期。
2. 不修改 C++ model card 或 MTP forward 接入。
3. 不补 L1/CUDA 算子能力；W8 linear、concat、GQA append-batch 等算子准备工作拆到
   [Part 2](2026-07-02-mtp-foundation-part2-operators.md)。
4. 不改变 `W8G128_F16S` 的量化公式、packing、qtype id 或 file header version。
5. 不引入 V4，不改 magic，不改 module enum，不改 source kind enum 数值。
6. 不改变 TEXT_CORE qtype assignment；TEXT_CORE 当前没有 W8 linears。
7. 不改变 VISION_ENCODER 的 `W8G128_F16S` merger FC assignment。

### 1.3 State Boundary: MTP 与 GDN State

MTP_DRAFT 不使用主模型的 `GdnState`，也不需要自己分配一套独立的 `GdnState`。本
checkpoint 的 MTP layer 是 full-attention decoder layer，不是主模型 48 个 linear/GDN
层，因此没有 GDN recurrent state 或 conv state。

状态边界如下：

- target prefill/decode 照常更新 target full-attention KV cache 和 target GDN state；
- MTP first proposal 读取 target final hidden states 和 token ids，写入 MTP 自己的
  full-attention KV namespace；
- MTP 不读写 target KV cache，不读写 target GDN state，也不把自己的 hidden state 写回
  target layer state；
- 多步 draft 时，下一步 MTP hidden input 来自上一 MTP final hidden，持久状态仍是 MTP KV
  和 MTP autoregressive hidden buffer，不是 GDN state；
- target verification 如果批量处理 draft suffix，后续 runtime 必须定义 target KV/GDN 的
  commit-or-rollback 语义，避免 rejected suffix 留在主流程状态里；这属于后续引擎状态生命周期，
  不属于本准备部分。

## 2. Current State

当前 q5090 v3 已经把 `MTP_DRAFT` 作为 first-class module 预留：

```text
ModuleKind::MtpDraft = 1
SourceKind::MtpFc = 50
SourceKind::MtpPreFcNormEmb = 51
SourceKind::MtpPreFcNormHid = 52
SourceKind::MtpNorm = 53
```

MTP attention / MLP / layernorm 权重复用 shared source kinds：

```text
INPUT_LAYERNORM
POST_ATTN_LAYERNORM
ATTN_Q / ATTN_GATE / ATTN_K / ATTN_V / ATTN_Q_NORM / ATTN_K_NORM / ATTN_O
MLP_GATE / MLP_UP / MLP_DOWN
```

当前 Python tensor plan 中的 MTP layout 已经是 v3 fused assignment：12 个 blocks、16 个
segments、2 个 fusion groups：

```text
mtp.fc.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.attn_in.w8
  mtp.layers.0.self_attn.q_proj.q
  mtp.layers.0.self_attn.k_proj.weight
  mtp.layers.0.self_attn.q_proj.gate
  mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.gateup.w8
  mtp.layers.0.mlp.gate_proj.weight
  mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
mtp.norm.weight
```

其中所有 MTP dense/fused linears 仍是 `W8G128_F16S` `ROW_SPLIT` `group_size=128`。layout 已经
满足 Part 1 的 fusion 目标，但 MTP 是 speculative draft 的精度敏感路径，接受率会直接受 draft
模型误差影响。对比本地 GGUF MTP artifact，MTP dense tensors 使用 GGUF `Q8_0`（32 元素一组
FP16 scale）；原始 HF MTP tensors 是 BF16。因此本阶段采用精度优先策略：MTP dense/fused
linears 使用 `W8G32_F16S`，MTP norms 仍使用原始 BF16 精度。

- raw `self_attn.q_proj.weight [12288,5120]` 同时包含 query 与 gate；
- `attn_in.w8` 和 `mlp.gateup.w8` 已经保留 runtime-native row ranges；
- 当前剩余问题是 MTP dense/fused linears 的 scale 粒度比 GGUF MTP `Q8_0` 更粗。

当前 `tools/parity/ref_model.py` 是 q5090 correctness-first Python oracle。它已经具备以下
基础能力：

- `Q5090File` 解析 module、block、segment、fusion group；
- `self.entries[name]` 按 block name 访问完整 block；
- `self.views[name]` 按 segment name 访问 projection view；
- `block_tensor(name)` 可以解码一个 fused block 的完整 `[N,K]` 矩阵；
- `tensor(name)` 可以解码 segment view，包括 fused block 内的 row slice；
- target 主干 forward 已实现 full-attention/GDN/MLP/final logits。

但它当前没有 MTP-specific forward：没有 MTP KV namespace，没有 `mtp_forward()`，也没有基于
`mtp.fc`、`mtp.attn_in.w8`、`mtp.mlp.gateup.w8` 的 draft logits 计算。Part 1 的 layout
修改应以补齐这条 Python ref path 作为收尾，证明新的 q5090 MTP layout 不只是结构可解析，
还可以按模型语义被正确消费。

## 3. Target MTP_DRAFT Weight Layout

### 3.1 Block Inventory

新的 MTP_DRAFT module 仍然只有一层 MTP layer，`source_layer=0`。全局 MTP tensors 使用
`source_layer=NO_LAYER`。

| block name | qtype/layout | shape `[N,K]` or dense shape | source kind / segments |
|---|---|---:|---|
| `mtp.fc.weight` | W8G32 row-split | `[5120,10240]` | `MTP_FC`, `NO_LAYER` |
| `mtp.pre_fc_norm_embedding.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_PRE_FC_NORM_EMB`, `NO_LAYER` |
| `mtp.pre_fc_norm_hidden.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_PRE_FC_NORM_HID`, `NO_LAYER` |
| `mtp.layers.0.input_layernorm.weight` | BF16_CTRL contiguous | `[5120]` | `INPUT_LAYERNORM`, layer 0 |
| `mtp.layers.0.attn_in.w8` | W8G32 row-split | `[14336,5120]` | block `OTHER`; segments `ATTN_Q/K/GATE/V`, layer 0 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16_CTRL contiguous | `[256]` | `ATTN_Q_NORM`, layer 0 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16_CTRL contiguous | `[256]` | `ATTN_K_NORM`, layer 0 |
| `mtp.layers.0.self_attn.o_proj.weight` | W8G32 row-split | `[5120,6144]` | `ATTN_O`, layer 0 |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16_CTRL contiguous | `[5120]` | `POST_ATTN_LAYERNORM`, layer 0 |
| `mtp.layers.0.mlp.gateup.w8` | W8G32 row-split | `[34816,5120]` | block `OTHER`; segments `MLP_GATE/MLP_UP`, layer 0 |
| `mtp.layers.0.mlp.down_proj.weight` | W8G32 row-split | `[5120,17408]` | `MLP_DOWN`, layer 0 |
| `mtp.norm.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_NORM`, `NO_LAYER` |

Block count becomes 12. Segment count becomes 16:

- 10 standalone single-segment blocks;
- 4 segments in `mtp.layers.0.attn_in.w8`;
- 2 segments in `mtp.layers.0.mlp.gateup.w8`.

### 3.2 `mtp.attn_in.w8` Segment Layout

MTP raw checkpoint has:

```text
mtp.layers.0.self_attn.q_proj.weight  [12288,5120]
mtp.layers.0.self_attn.k_proj.weight  [1024,5120]
mtp.layers.0.self_attn.v_proj.weight  [1024,5120]
```

The raw q projection is packed per head:

```text
q_proj.reshape(24, 2*256, 5120)
  [:, 0:256, :]   -> query
  [:, 256:512, :] -> gate
```

Equivalent row-index formula:

```text
query[h*256 + d, :] = raw_q_proj[h*512 + d, :]
gate[h*256 + d, :]  = raw_q_proj[h*512 + 256 + d, :]

where h in [0,24), d in [0,256)
```

The converter must reuse the same de-interleave transform as TEXT_CORE full-attention q_proj.
A naive contiguous `[0:6144] / [6144:12288]` split is incorrect because query/gate are interleaved
inside every head.

Target fused block:

```text
name:           mtp.layers.0.attn_in.w8
module_kind:    MTP_DRAFT
qtype:          W8G32_F16S
layout:         ROW_SPLIT
group_size:     32
scale_dtype:    FP16
shape:          [14336,5120]
source_layer:   0
source_kind:    OTHER
fusion_group:   ATTN_IN
fusion_index:   0
```

Segments:

| row range | rows | segment name | source kind | source tensor / transform |
|---:|---:|---|---|---|
| `[0,6144)` | 6144 | `mtp.layers.0.self_attn.q_proj.q` | `ATTN_Q` | `q_proj.weight` with query transform |
| `[6144,7168)` | 1024 | `mtp.layers.0.self_attn.k_proj.weight` | `ATTN_K` | direct |
| `[7168,13312)` | 6144 | `mtp.layers.0.self_attn.q_proj.gate` | `ATTN_GATE` | `q_proj.weight` with gate transform |
| `[13312,14336)` | 1024 | `mtp.layers.0.self_attn.v_proj.weight` | `ATTN_V` | direct |

Runtime output layout after one W8 linear:

```text
attn_in_out [14336,T]

q    = attn_in_out[0:6144,      :].view([256,24,T])
k    = attn_in_out[6144:7168,   :].view([256, 4,T])
gate = attn_in_out[7168:13312,  :].view([256,24,T])
v    = attn_in_out[13312:14336, :].view([256, 4,T])
```

This order deliberately keeps q/k first and gate/v second. It mirrors the current TEXT_CORE runtime
pattern, except MTP uses a single W8G32 block rather than separate Q4 and Q5 blocks.

### 3.3 `mtp.mlp.gateup.w8` Segment Layout

Target fused block:

```text
name:           mtp.layers.0.mlp.gateup.w8
module_kind:    MTP_DRAFT
qtype:          W8G32_F16S
layout:         ROW_SPLIT
group_size:     32
scale_dtype:    FP16
shape:          [34816,5120]
source_layer:   0
source_kind:    OTHER
fusion_group:   MLP_GATEUP
fusion_index:   0
```

Segments:

| row range | rows | segment name | source kind | source tensor |
|---:|---:|---|---|---|
| `[0,17408)` | 17408 | `mtp.layers.0.mlp.gate_proj.weight` | `MLP_GATE` | direct |
| `[17408,34816)` | 17408 | `mtp.layers.0.mlp.up_proj.weight` | `MLP_UP` | direct |

Runtime output layout:

```text
gateup [34816,T]

gate = gateup[0:17408,     :]
up   = gateup[17408:34816, :]
act  = silu(gate) * up      # [17408,T]
```

### 3.4 Standalone W8 Blocks

These remain standalone because they consume different inputs or are not worth fusing with unrelated
work:

| block name | shape | reason |
|---|---:|---|
| `mtp.fc.weight` | `[5120,10240]` | consumes packed `[embedding_norm; hidden_norm]` input |
| `mtp.layers.0.self_attn.o_proj.weight` | `[5120,6144]` | consumes attention output after gate multiply |
| `mtp.layers.0.mlp.down_proj.weight` | `[5120,17408]` | consumes SwiGLU activation |

## 4. q5090 v3 Spec Impact

No binary version changes are required.

The following remain unchanged:

- file magic and version;
- `ModuleKind` numeric values;
- `SourceKind` numeric values;
- qtype id `W8G128_F16S = 3`;
- `W8G128_F16S` quantization formula;
- W8 row-split payload layout: one signed int8 code per K element, no high plane, FP16 scale per
  `(row, group)`;
- `W8G128_F16S` remains `group_size=128`, `scale_dtype=FP16`, `high_plane_bytes=0`;
- module ordering TEXT -> MTP -> VISION.

The following is new in v3 for this phase:

- qtype id `W8G32_F16S = 6`;
- `W8G32_F16S` uses the same signed int8 code formula as `W8G128_F16S`, but with `group_size=32`;
- only MTP_DRAFT dense/fused linears use `W8G32_F16S`;
- MTP_DRAFT norms remain `BF16_CTRL`;
- TEXT_CORE remains unchanged and has no W8 tensors;
- VISION_ENCODER keeps its existing `W8G128_F16S` merger FC tensors.

The only normative document edited in this phase is:

1. `docs/q5090_packed_file_format_v3.md`
   - Replace the current v3 wording that says MTP fusion is deferred or standalone.
   - Stop saying the v2 tensor plan applies to v3 unchanged for MTP_DRAFT. The v3 spec should own the
     new MTP_DRAFT tensor assignment directly.
   - Add a v3-owned MTP_DRAFT fused assignment: 12 blocks, 16 segments, and two MTP fusion groups.
   - State explicitly that `docs/qwen3_6_27b_q5090_v2_tensor_plan.md` and
     `docs/qwen3_6_27b_q5090_final_quant_format_v1.md` are not edited by this change.
   - Keep `W8G128_F16S` numeric policy unchanged.
   - Add `W8G32_F16S` as a v3 qtype and assign it only to MTP dense/fused linears.
   - The v3 spec owns the new MTP block/segment/fusion assignment.

No file version bump is introduced by this layout and MTP-precision change. It changes MTP block
grouping, source transforms, and MTP dense qtype assignment inside v3; it does not redefine existing
qtype semantics.

Fusion ids are reused rather than extended:

```text
FUSION_ATTN_IN    = 1
FUSION_MLP_GATEUP = 3
```

No MTP-specific fusion id is needed. `FusionGroupRecord` remains module-agnostic; its members are
identified by `first_block_tensor_index`, and each member block carries
`TensorEntry.module_kind = MTP_DRAFT`. Fused lookup is disambiguated by
`(module_kind, fusion_group_id, fusion_index, source_layer)`. Segment/weight lookup is disambiguated by
`(module_kind, source_kind, source_layer)`. For the fused MTP blocks themselves,
`TensorEntry.source_kind = OTHER`; the projection identity is carried only by the segments.

For a full artifact that currently counts MTP as 15 standalone blocks / 15 segments / 0 MTP fusion
groups, this layout changes the full artifact counts by:

```text
block_count:        -3
segment_count:      +1
fusion_group_count: +2
```

For the current full-artifact baseline `1167 blocks / 1311 segments / 128 fusion groups`, the expected
new structural counts are `1164 blocks / 1312 segments / 130 fusion groups`.

## 5. Python Converter And Verifier Adaptation

### 5.1 Tensor Plan Generation

Current `tools/q5090_convert/tensor_plan.py::build_mtp_specs()` returns standalone `TensorSpec`
entries. Fused MTP requires block + segment + fusion-group construction, like TEXT_CORE.

Recommended Python shape:

```text
build_mtp_manifest() -> ExpectedManifest
```

or an append-style helper:

```text
append_mtp_blocks(blocks, segments, fusion_groups)
```

The helper should emit:

```text
standalone mtp.fc
standalone pre_fc_norm_embedding
standalone pre_fc_norm_hidden
standalone input_layernorm
fused mtp.layers.0.attn_in.w8
standalone q_norm
standalone k_norm
standalone o_proj
standalone post_attention_layernorm
fused mtp.layers.0.mlp.gateup.w8
standalone down_proj
standalone mtp.norm
```

The MTP q/gate extraction must call the same transform logic as TEXT_CORE:

```text
TRANSFORM_ATTN_QPROJ_QUERY
TRANSFORM_ATTN_QPROJ_GATE
```

This is value-defining. The transformed q and gate segments must be quantized independently into their
own row ranges inside `mtp.layers.0.attn_in.w8`.

### 5.2 Converter Assembly

Current conversion appends MTP as standalone specs. The conversion path needs to:

1. create the MTP module begin index after TEXT_CORE;
2. append fused and standalone MTP blocks in canonical order;
3. append fusion group records for MTP `ATTN_IN` and `MLP_GATEUP`;
4. include MTP_DRAFT bytes in the full q5090 artifact, without changing runtime load-policy or module
   lifecycle behavior in this phase;
5. keep module ordering TEXT -> MTP -> VISION.

The converter output must be consumable by the Python ref model in this phase. That means the artifact
must preserve enough block/segment/fusion metadata for `Q5090File` to resolve both full fused blocks
(`mtp.layers.0.attn_in.w8`, `mtp.layers.0.mlp.gateup.w8`) and their logical projection segments. The
converter still does not implement model execution; it transforms, quantizes, packs, writes, and
structurally verifies the new MTP layout.

### 5.3 Verifier Updates

`tools/q5090_convert/verify.py` should validate:

- MTP module exists when the full artifact is expected;
- MTP block count and segment count match the new manifest;
- fused MTP blocks have block-level `source_kind == OTHER`;
- `mtp.layers.0.attn_in.w8` has shape `[14336,5120]`;
- `mtp.layers.0.attn_in.w8` row ranges exactly partition `[0,14336)`;
- q/gate segments reference the raw `q_proj.weight` source with the correct transforms;
- `mtp.layers.0.mlp.gateup.w8` has shape `[34816,5120]`;
- gate/up row ranges exactly partition `[0,34816)`;
- MTP fusion groups reuse ids `ATTN_IN=1` and `MLP_GATEUP=3`, with member
  `TensorEntry.module_kind == MTP_DRAFT`;
- exactly the five MTP dense/fused blocks are `W8G32_F16S` with `group_size=32`,
  `scale_dtype=FP16`, and `high_plane_bytes=0`;
- all MTP norms remain `BF16_CTRL` contiguous;
- TEXT_CORE has no W8G32 or W8G128 blocks;
- VISION_ENCODER keeps its two merger FC blocks as `W8G128_F16S`.

Packing tests must cover W8G32 encode/decode in addition to existing W8G128 coverage. Converter-level
structural tests must change because the expected MTP manifest and qtype assignment change.

## 6. Python Ref Model MTP Loading

The ref model should consume the q5090 file exactly as the C++ loader is expected to consume it later:
through q5090 block/segment metadata, not by reconstructing checkpoint names outside the file.

Required q5090 lookup behavior:

| logical use | q5090 lookup | expected tensor |
|---|---|---|
| MTP fc | `weight("mtp.fc.weight")` or equivalent segment view | `[5120,10240]` |
| MTP attention input | `block_weight("mtp.layers.0.attn_in.w8")` | `[14336,5120]` |
| MTP q segment, optional debug | `tensor("mtp.layers.0.self_attn.q_proj.q")` | `[6144,5120]` |
| MTP k segment, optional debug | `tensor("mtp.layers.0.self_attn.k_proj.weight")` | `[1024,5120]` |
| MTP gate segment, optional debug | `tensor("mtp.layers.0.self_attn.q_proj.gate")` | `[6144,5120]` |
| MTP v segment, optional debug | `tensor("mtp.layers.0.self_attn.v_proj.weight")` | `[1024,5120]` |
| MTP gate/up | `block_weight("mtp.layers.0.mlp.gateup.w8")` | `[34816,5120]` |
| shared embedding | `row_split_rows("model.language_model.embed_tokens.weight", ids)` | `[T,5120]` |
| shared lm_head | `row_split_row_chunks("lm_head.weight", ...)` | chunked `[rows,5120]` |

The full fused block lookups are the path used by the MTP forward. Segment lookups are still useful for
structural dumps and one-off parity probes, but the ref forward should exercise the fused layout because
that is the layout Part 1 is validating.

`Q5090File` already has most of the required mechanics:

- `self.entries` indexes block names;
- `self.views` indexes segment names;
- `_decode_view()` can slice row ranges out of a fused `ROW_SPLIT` block;
- `block_tensor()` can decode an entire fused block;
- `row_split_rows()` and `row_split_row_chunks()` already support shared embedding and lm_head access.

The required ref-model work is therefore not a new parser abstraction. It is MTP-specific binding and
forward logic on top of the existing q5090 reader.

## 7. Python Ref Model MTP Forward

The ref model should expose an MTP entry point with explicit inputs:

```text
mtp_forward(input_ids:     [T],
            hidden_states: [T,5120],
            positions:     [T],
            phase:         "prefill" | "decode")
  -> mtp_hidden: [T,5120]
  -> logits_last or logits: [248320] or [T,248320]
  -> draft_token: int, for greedy smoke paths
```

`hidden_states` must be the target hidden representation that would be sent to target `lm_head`. In the
current Python ref model, target `run_layers()` returns the residual stream before final norm, while
`logits_last()` applies `model.language_model.norm.weight`. A caller that wants to feed target output into
MTP must therefore pass the post-final-norm hidden:

```text
target_hidden_for_mtp = rmsnorm(run_layers_out,
                                model.language_model.norm.weight,
                                unit_offset=True)
```

For additional autoregressive MTP steps, `hidden_states` is the previous `mtp_hidden`, not target GDN
state and not target residual stream.

### 7.1 MTP Input Projection

MTP reuses the shared text embedding and applies the two pre-fc norms:

```text
emb = embed(input_ids)                                               # [T,5120]
e   = rmsnorm(emb,        mtp.pre_fc_norm_embedding.weight, true)    # [T,5120]
h   = rmsnorm(hidden_in,  mtp.pre_fc_norm_hidden.weight,    true)    # [T,5120]
u   = cat([e, h], dim=-1)                                            # [T,10240]
x   = linear(u, mtp.fc.weight)                                       # [T,5120]
```

All MTP RMSNorms use the same `(1+w)` convention as target layer norms and q/k norms. The only known
plain `w` norm remains the target GDN gated norm, which MTP does not use.

### 7.2 MTP Full-Attention Layer

The single MTP layer consumes `x` through the fused W8 attention input block:

```text
h        = rmsnorm(x, mtp.layers.0.input_layernorm.weight, true)     # [T,5120]
attn_in  = linear(h, mtp.layers.0.attn_in.w8)                        # [T,14336]

q        = attn_in[:, 0:6144].reshape(T,24,256)
k        = attn_in[:, 6144:7168].reshape(T,4,256)
gate     = attn_in[:, 7168:13312].reshape(T,24,256)
v        = attn_in[:, 13312:14336].reshape(T,4,256)

q_norm   = rmsnorm(q, mtp.layers.0.self_attn.q_norm.weight, true)
k_norm   = rmsnorm(k, mtp.layers.0.self_attn.k_norm.weight, true)
q_rope, k_rope = apply_rope(q_norm, k_norm, positions)
attn     = mtp_gqa_attention(q_rope, k_rope, v, phase)
attn     = sigmoid_gate_mul(gate, attn).reshape(T,6144)
o        = linear(attn, mtp.layers.0.self_attn.o_proj.weight)
x        = residual_add(x, o)
```

`mtp_gqa_attention()` must use a separate MTP KV namespace, for example `self.mtp_kv[0]`, rather than
the target full-attention `self.kv[fidx]`. It does not use `self.conv` or `self.ssm`.

The MTP MLP then consumes the fused W8 gate/up block:

```text
h       = rmsnorm(x, mtp.layers.0.post_attention_layernorm.weight, true)
gateup  = linear(h, mtp.layers.0.mlp.gateup.w8)                     # [T,34816]
gate    = gateup[:, 0:17408]
up      = gateup[:, 17408:34816]
act     = silu_and_mul(gate, up)
down    = linear(act, mtp.layers.0.mlp.down_proj.weight)
x       = residual_add(x, down)
```

Final MTP output and logits:

```text
mtp_hidden = rmsnorm(x, mtp.norm.weight, true)                      # [T,5120]
logits     = mtp_hidden @ lm_head.weight.T                          # [T,248320]
draft      = argmax(logits[-1])
```

The logits path should reuse the existing chunked `lm_head.weight` reader so the ref model does not need
to materialize the full `[248320,5120]` matrix unless the caller explicitly requests it.

### 7.3 Ref Model State Boundary

`reset_state()` should reset both target and MTP state:

```text
self.kv      # target full-attention KV, indexed by target full layer index
self.mtp_kv  # MTP full-attention KV, indexed by MTP layer index; only 0 for this checkpoint
self.conv    # target GDN conv state
self.ssm     # target GDN recurrent state
```

MTP must never read or write `self.conv` / `self.ssm`. It also must not insert MTP K/V into target
`self.kv`. A simple separate dictionary is enough for the Python oracle; this is not a C++ runtime
allocation policy.

For `phase="prefill"`, MTP attention may replace `self.mtp_kv[0]` with the supplied sequence K/V and use
a causal mask over `T`. For `phase="decode"`, it appends the single or small current K/V batch to
`self.mtp_kv[0]` and attends over the full MTP prefix. This mirrors the existing target ref attention
helper, but under a separate namespace.

## 8. Acceptance Criteria For This Preparation Part

This preparation part is complete when:

1. `docs/q5090_packed_file_format_v3.md` describes the new MTP_DRAFT fused layout without a version
   bump, including `W8G32_F16S` and its MTP-only assignment; no q5090 v1/v2 document is edited for this
   change.
2. Python converter emits MTP_DRAFT with 12 blocks and 16 segments in the layout above, with every MTP
   dense/fused linear stored as `W8G32_F16S`.
3. Python verifier rejects the old standalone MTP layout when the new full artifact is expected.
4. Python verifier checks fused MTP block-level `source_kind == OTHER`, MTP segment row ranges, and MTP
   fusion ids `ATTN_IN=1` / `MLP_GATEUP=3`; it also enforces MTP-only `W8G32_F16S`, no W8 in
   TEXT_CORE, and VISION merger FC remaining `W8G128_F16S`.
5. `tools/parity/ref_model.py` can load the new MTP_DRAFT fused blocks and resolve their segment views.
6. `tools/parity/ref_model.py` implements an MTP forward path that consumes `input_ids`,
   target-final-norm hidden states, positions, and the separate MTP KV namespace.
7. A greedy ref-model smoke path can produce at least one MTP draft token through shared
   `embed_tokens` and shared `lm_head.weight`.

Only after these layout/ref-model pieces exist should Part 2 or later runtime work consume the MTP
weights from L1/C++ code. Part 2 owns W8 linear kernels, concat/pack, GQA append-batch behavior, and
operator-level shape coverage.
