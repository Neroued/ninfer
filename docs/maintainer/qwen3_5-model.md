# Qwen3.5 model reference

This reference describes the Qwen3.5 Dense mathematics implemented by NInfer, including
Text, MTP, Vision and their state semantics. The Qwen3.6 and Qwen3.8 releases used by the official
artifacts are instances of this architecture. Their release names do not choose another graph.

[Engine architecture](engine-architecture.md) defines ownership and execution;
[artifact container](artifact-container.md) defines stored objects and bindings;
[weight conversion](../weight-conversion.md) explains source mapping and recipes.

## Architecture and configuration

The architecture registry recognizes these Text config pairs:

| `architectures[0]` | `model_type` | Feed-forward block |
|---|---|---|
| `Qwen3_5ForCausalLM` | `qwen3_5_text` | Dense SwiGLU |

Code fixes the pre-norm residual order, gated GQA, zero-centered Text norms, GDN recurrence,
interleaved partial MRoPE, FFN mathematics and component relationships. Config supplies instance
dimensions and layer choices. It does not describe a graph of arbitrary operators.

[`config.cpp`](../../src/models/qwen3_5/config.cpp) defines the exact fields and validation.
Their responsibilities are:

| Config facts | Meaning |
|---|---|
| `hidden_size`, `vocab_size`, `num_hidden_layers`, `max_position_embeddings` | Mathematical extents |
| `layer_types` | One `full_attention` or `linear_attention` entry per Text block |
| `tie_word_embeddings`, `rms_norm_eps` | Training relationship and normalization constant |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | GQA dimensions; also used by MTP |
| `rope_parameters` | Theta, partial rotary factor and three interleaved MRoPE sections |
| `linear_num_*_heads`, `linear_*_head_dim`, `linear_conv_kernel_dim` | GDN heads, state dimensions and convolution width |
| `intermediate_size` | Dense FFN width |

Projection widths, compact attention/GDN layer indices, rotary dimensions and parameter shapes are
derived. The converter expands source shorthand such as a full-attention interval into the explicit
`layer_types` array. Source dtype, quantization metadata, checkpoint name and training-only fields
do not become mathematical config.

Text is required. Optional components target `text` and supply their private data:

- Vision supplies tower dimensions and patch/merge geometry. Its output width comes from Text.
- MTP identifies `Qwen3_5MTP`. The single-layer predictor inherits Text's
  attention, RoPE and FFN geometry and shares its embedding/output semantics.
- DFlash2 supplies private draft dimensions, attention pattern, conditioning taps and its
  algorithm parameters. See [DFlash2](dflash.md).

Only selected components are semantically bound and materialized. A model's weights and config
remain immutable after loading; request capacity, concurrency, KV codecs and graph profiles are
runtime choices.

### Current official geometries

These are useful workload points, rather than a second config registry:

| Quantity | 27B Dense |
|---|---:|
| Hidden width H | 5120 |
| Text layers | 64 |
| Full attention / GDN layers | 16 / 48 |
| Q heads / KV heads / head dimension | 24 / 4 / 256 |
| GDN key heads / value heads | 16 / 48 |
| GDN key/value head dimension | 128 / 128 |
| GDN convolution width | 4 |
| Dense intermediate width | 17408 |
| Embedding/output matrix rows | 248320 |
| Position capacity | 262144 |

It uses `rms_norm_eps=1e-6`, `rope_theta=1e7`, `partial_rotary_factor=0.25` and
`mrope_section=[11,11,10]`. Its full-attention layers occur at indices `3,7,...`; that pattern is
stored in config. It has untied embedding and output matrices. Qwen3.6-27B and Qwen3.8-27B share
this Dense geometry, while training results, representations and frontend semantics can differ.

Config parsing accepts structural variation. Actual execution remains bounded by implemented
Ops: the current fused input projections have explicit shapes and GDN uses the implemented
128-dimensional recurrence.
The current multimodal RoPE entry requires the implemented interleaved axis mapping. These checks
belong to the relevant consumers; parsing a config is not a promise to execute every size.

## Logical parameters and physical bindings

[`load/`](../../src/models/qwen3_5/load/) expands the selected config into parameter names, shapes
and mathematical input Uses. The converter's
[`qwen3_5.py`](../../tools/convert/qwen3_5.py) maps source tensors into the same roles.

The main namespaces are `text/`, `vision/`, `mtp/`, `dflash2/` and the optional `proposal/`. Text
block `i` contains `text/layers/i/attention/...` or `.../gdn/...`, plus its Dense `mlp/...`
parameters. A projection such as `attention/query` has logical shape
`[num_attention_heads * head_dim, H]` and a Use at `text/layers/i/mixer_input`.

Bindings describe the logical values in physical parents. They can use complete objects,
consecutive regions or several ordered parts. The source `q_proj` stores each head's Q rows next
to that head's gate rows; source mapping extracts those rows into separate logical projections.
A simple split at the middle of the source matrix would change the model.

The Dense groupwise attention recipe packs Q/K into Q4 and gate/V into Q5. A supported FP8 or NVFP4
recipe can pack their union into one parent. The same four logical roles then become either the
two-weight or one-weight native fused call. Dense gate/up follows its finite packing and execution
rules. [Storage layouts](storage-layouts.md#8-logical-views-and-native-operands)
describes the difference between logical views and native operand restrictions.

`tie_word_embeddings` records a training relationship; bindings choose whether physical data is
shared. MTP and draft components reference Text's embedding/head rather than introduce private
copies by default. Shared weights keep independent Uses and activation permissions. The stored
NVFP4 weight divisor belongs to its parent; an activation divisor belongs to a mathematical input.

## Text block mathematics

In the formulas below, activations are written with feature dimensions first and token columns
last. H is hidden width, D is an attention head dimension, and T is the current token extent.
The two Text norm forms are:

```text
offset_rmsnorm(x,w) = (1 + w) * x / sqrt(mean(x^2) + eps)
plain_rmsnorm(x,w)  =       w  * x / sqrt(mean(x^2) + eps)
```

Text/MTP input, post-attention, final, stem and Q/K norms use the offset form. GDN's internal
output norm uses the plain form. Vision uses LayerNorm with weight and bias.

Every Text block follows:

```text
h = offset_rmsnorm(x, input_norm)
x = x + mixer(h)
h = offset_rmsnorm(x, post_attention_norm)
x = x + ffn(h)
```

The mixer is gated GQA or GDN according to `layer_types`. Dense FFN is
`down(SiLU(gate(h)) * up(h))`. Text/MTP projections and GDN convolution are
bias-free. Fusion may combine projection, activation, normalization and residual work while
preserving the declared mathematical and observable precision boundaries.

### Gated full attention

```text
q    = query(h)
gate = gate_projection(h)
k    = key(h)
v    = value(h)

q = offset_rmsnorm(q, query_norm)
k = offset_rmsnorm(k, key_norm)
q, k = partial_interleaved_mrope(q, k)
a = causal_gqa(q, k, v, scale=1/sqrt(D))
y = output(a * sigmoid(gate))
```

Q and gate have one D-dimensional vector per query head. K and V have one per KV head, with each
KV head serving a consecutive group of query heads. The gate is neither normalized nor rotated.
The logical cached values are normalized/rotated K and projected V. Prefill appends a chunk and
uses causal attention; decode appends one column and attends over the visible prefix.

Paging and KV quantization are storage choices. Their contracts are
[`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h),
[`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) and
[paged KV cache](paged-kv-cache.md).

### Gated DeltaNet

Normalized input produces Q, K, V, output gate Z and per-value-head A/B projections. Only Q/K/V
pass through depthwise causal convolution and SiLU. The convolution retains the preceding
`linear_conv_kernel_dim - 1` projected columns. GDN has no RoPE or position-ID input.

For value head j, use Q/K head `j // (value_heads / key_heads)`. Let the recurrent state be
`S[Dv,Dk]`. One token performs:

```text
g     = -exp(A_log) * softplus(a + dt_bias)
alpha = exp(g)
beta  = sigmoid(b)
k     = k / sqrt(sum(k^2) + 1e-6)
q     = q / sqrt(sum(q^2) + 1e-6)
Sbar  = alpha * S
u     = beta * (v - Sbar @ k)
S     = Sbar + u outer k
o     = (S @ q) * (1/sqrt(Dk))
y     = out_projection(plain_rmsnorm(o, gdn_norm) * SiLU(z))
```

The delta error uses the already-decayed state. GDN controls and recurrent state are FP32. The Op
receives raw BF16 q/k; normalization precision and the chunked/recurrent implementation are private
to its qualified route. Large prefill can use parallel state passing, while decode uses recurrence.

Ordinary decode updates the lane's current state. Speculative verification leaves persistent state
unchanged and writes convolution and recurrent records. After output resolution, one Fold applies
the committed prefix of that recorded physical block. The projection/convolution/record fusion
remains in the model's finite execution choices. [ReplaySSM](replayssm-gdn.md) defines the record
and commit semantics, including their numerical boundary.

## Prefill, decode and MTP

Text prefill gathers embedding columns, replaces media placeholders with Vision outputs, and runs
the configured blocks in chunks. It preserves state and positions across chunks. The blocks run in
the Text forward's plain residual ([Qwen4Exp reference, the Text forward's
options](qwen4-exp-model.md#the-text-forwards-options); `execution/qwen4_forward.h`), with the
27B's fused Ops in every block. Final offset
RMSNorm and the full output head produce the target distribution. Ordinary decode repeats this
schedule for one new token. CUDA Graph replay uses the same selected model implementation.

MTP is a one-layer predictor conditioned on a final-normalized target hidden state h_t and the
following token x_(t+1):

```text
e = offset_rmsnorm(composed_embedding(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t, pre_fc_norm_hidden)
u = input_projection(concat(e, h))
u = one_gated_attention_block_with_target_ffn(u)
draft_hidden = offset_rmsnorm(u, mtp_final_norm)
draft_logits = target_output_head(draft_hidden)
```

The stem concatenates embedding first, hidden second. MTP has private stem, block, final norm and
KV state. Its attention and FFN geometry come from Text. It can recursively propose several
tokens using the previous MTP hidden output.

For prompt tokens x_0 through x_n and the target's first generated token x_(n+1), MTP prefill uses:

```text
token inputs    = [x_1, x_2, ..., x_n, x_(n+1)]
hidden inputs   = [h_0, h_1, ..., h_(n-1), h_n]
position inputs = [p_0, p_1, ..., p_(n-1), p_n]
```

Tokens shift; hidden states and positions do not. The whole aligned sequence populates MTP KV.
Shifted media placeholders use their corresponding composed Vision embedding columns.
[`mtp_alignment.h`](../../src/models/qwen3_5/program/speculative/mtp_alignment.h) and the prefill
implementation own chunk-boundary and continuation alignment.

An optional proposal head supplies an indexed vocabulary subset for draft prediction. Its row map
converts proposal rows to actual token IDs. Full target verification continues to use the full
output head. Backend selection, draft width and proposal-head choice are fixed at startup.

## Tensor-parallel execution

A Dense model loaded with `LoadOptions.tp=2` is split across two devices by heads, intermediate
width and vocabulary ([`sharding.h`](../../src/models/qwen3_5/load/sharding.h)); rank r holds
`execution::Parameters(model, r)`. The hidden/residual axis is replicated. `TextContext` runs over
both ranks when it is given a `TpExecution` ([`tp.h`](../../src/models/qwen3_5/execution/tp.h))
naming rank 1's parameters, arena, GDN state pool, KV caches, prefill KV rows, ordinary decode
control, Text forward runtime, the ReplaySSM records of a speculative backend and, under MTP, its
prefill frames; the Text layers run in the Qwen4 forward on both ranks.

Each block issues one rank's work on that device's stream:

```text
h_r      = offset_rmsnorm(x, input_norm)                 replicated
attention: q|k|gate|v_r = column_parallel(h_r)          12 query / 2 KV heads per rank (27B)
           a_r = gated head-local attention over rank r's KV heads and pages
GDN:       g|beta_r = column_parallel(h_r); q|k|v|z_r = column_parallel(h_r)
           a_r = conv + recurrence over rank r's 8 key / 24 value heads and state slots
x       += all_reduce(row_parallel(a_r))                 residual added once, by rank 0
x       += all_reduce(row_parallel(swiglu(column_parallel(offset_rmsnorm(x, post_norm)))))
```

Both all-reduces leave the identical BF16 sum on the two ranks, so every later per-rank input
(norms, KV pages, GDN state) agrees without further exchange. The GDN input norm runs separately
before the split gating projection because the fused norm-and-gating Op has no split form. The
final norm is replicated; each rank projects its half of the vocabulary, and rank 0 alone
assembles the complete logits, where sampling runs: one cross-device copy pulls rank 1's
contiguous `[V/2, C]` half into rank-0 staging and one kernel (`concat_rows`) interleaves both
halves column by column (a captured 2D memcpy node could not follow the column count across graph
profiles). Rank 1 keeps no copy of the logits, in prefill, ordinary decode, verification
and the MTP proposals alike.

Rank 1 receives its own copies of the control tensors: prefill fills its positions on device and
reads its KV row from `TpExecution::text_kv_table_row`; ordinary decode reads
`TpExecution::ordinary`, which the Program uploads from the same host ingress record as rank 0's.
KV page and row bookkeeping stay on rank 0 and are mirrored to rank 1 at the same indices.

The split path covers text and multimodal prefill, ordinary decode, the MTP and DFlash2 rounds and
their logits. It rejects paired (two-parent) input projections and KV caches
other than BF16, INT8-G64 and NVFP4-G16, for which the 12/2-head attention has no route. RoPE has no per-rank
override.

MTP runs the same pattern over its one layer, with the MTP's own shards and KV pages:

```text
rank 0: e = offset_rmsnorm(embed(ids), embedding_norm)
rank 1: h = offset_rmsnorm(target_hidden, hidden_norm)
x       = all_reduce(fc_0 e + fc_1 h)                    fc split by input columns
q|k|gate|v_r = mtp_split_attn_in(packed_r offset_rmsnorm(x, input_norm))
x      += all_reduce(o_r gated_attention_r)              plain linear, then the residual add
x      += all_reduce(down_r silu(gate_r) * up_r)         gate|up split by rows
out     = offset_rmsnorm(x, final_norm)                  replicated
```

Rank 0's column half of the input projection contracts the normalized embedding and rank 1's the
normalized hidden, so the packed input is never formed and only rank 0 embeds tokens. The
attention input is the packed Q|K|Gate|V shard parent on every call (a rank's
`MtpProjectionParameters::packed`, [7168,5120] for the 27B), split by `mtp_split_attn_in`; the
separate `rows` projections the single-device prompt path uses are not read, so the KV-only and
query-only prompt projections compute and discard the other sections. The full proposal head is
the vocabulary-split output head, gathered before rank 0's argmax; the optimized proposal head is
loaded on rank 0 only and its proposal runs there, so rank 1's `MtpParameters::output_head` is empty
and unused. The Q8 MTP weights of the official artifacts use the registered Q8 halves
(`[5120,5120]`, `[7168,5120]`, `[5120,3072]`, `[17408,5120]`, `[5120,8704]`); a recipe that stores
the MLP and attention output as NVFP4 (`_mtp_nvfp4_layer`) runs `[5120,3072]`, `[17408,5120]` and
`[5120,8704]` on the NVFP4 `linear` halves the target registers, with 16-bit activations.

Verification runs the Text layers over K+1 columns with masked columns and records the ReplaySSM
inputs of each rank's own GDN heads (the 8/24-head fold geometry); the Program folds the accepted
prefix into both ranks' state with the same rows. Only rank 0 computes the target argmax and the
acceptance. Rank 1 receives the accepted counts, anchors, frontiers and licensed counts by
device-to-device copies ordered by a cross-device event, and derives everything else from its
own upload of the same round ingress with the same Ops. The prompt MTP alignment runs on both
ranks from each rank's final-normed chunk; rank 1 keeps its whole chunk in its own
`prefill_hidden`.

Under MTP rank 1 also retains its own copy of every target hidden the Program retains: the
prompt and forced-token tails (`copy_tail`), the hidden of a capture frontier inside a chunk
(written to the slot `PrefillContext::peer_rewrite_checkpoint_hidden` names), and the accepted
column of a verification round (`target_verify_accept`) or of a partial terminal commit. Each is
a local copy from rank 1's own replicated hidden into the same StateImage slot of its mirror
pool, so checkpoint Forks, Moves and copies carry it with rank 0's. The MTP bridge of a resumed
prefix runs the split head from each rank's retained copy (`TextContext::mtp_forward_batch` over
rank arrays) and appends both ranks' MTP K/V at the bridge position. A zero-suffix reuse samples
its first token through the vocabulary-split head from rank 0's retained hidden, which rank 1
pulls across devices once, so it needs no rank 1 copy and works for every backend. Rank 1 keeps
its own copy of the per-sequence RoPE delta (`TpExecution::rope_delta`): start_sequence publishes
it with rank 0's and every prefill chunk rewrites both, and the prompt proposal steps offset each
rank's positions by its own copy.

Vision is not split. `vision/*` is placed whole on `LoadOptions::vision_rank` (the index of
`EngineOptions::vision_device` in `devices`, rank 0 by default), so only that rank's Parameters
hold `vision`. The encoding rank's workspace follows the single-device Vision plan (general
prefix, encode region, item-output handoff); the other rank's (`WorkspacePlan::vision_receiver`,
`VisionContext::plan_receiver`) is its general prefix followed by a handoff of the same extent,
with no encode region. The per-rank reservation carries the encoding rank's workspace, and the KV
resolution credits the other rank with the difference
(`SequencePlanner::unallocated_reservation_bytes`). `VisionPrefillSession` encodes an item once,
on the encoding rank's stream; the other rank's stream waits on an event for the encode, copies
the `[H,V]` merged embeddings into its own handoff (`cudaMemcpyAsync`, staged through host memory
without peer access), and records a second event the encoding rank's stream waits on before it may
overwrite its handoff. The `VisionChunk` names both copies. The split multimodal prefill scatters each rank's copy
into its own residual at the chunk's visual columns (the hidden axis is replicated, so both ranks
hold the same embeddings) and uploads the three-axis M-RoPE positions to both ranks. Under MTP the
shifted visual overlap is scattered into rank 0's composed input embedding, the only embedding the
split MTP stem reads; the prompt MTP head, its final column and the prefix-reuse bridge accept
`[T,3]` positions, and a visual bridge column passes its composed embedding to rank 0. A DFlash2
tap reads rank 0's residual as for text.

At tp 2 DFlash2 splits the drafter and the candidate ranking of the optimized proposal head as
well as the target; placement, round and state are in [DFlash2](dflash.md#tensor-parallel-execution).

The Program plans one per-rank layout (`SequencePlanImpl::tp`): KV heads and GDN state halved,
and a workspace sized from the tp 2 workspace plan's allocation order with the rank's
`shard_text_config` extents, the all-reduce staging and the vocabulary-split logits; the ReplaySSM
records hold one rank's heads. Rank 1 allocates the same layout without the drafter's state
(`SequencePlanImpl::peer_persistent`: no DFlash context or features and no StateImage DFlash
rings); its Text and MTP KV pages, execution tables and StateImages are mirrors of rank 0's. Every
`ExecutionCore` carries the `TpExecution` (a prefill call's copy names the sequence's rank 1 MTP
row), so prompt prefill, forced tokens, ordinary decode, MTP and DFlash2 rounds (eager and
captured as one two-device graph) run on both ranks. The per-device CUDA Graph allowance at tp 2
(`tp2_*_graph_*allowance_bytes` in [`core/tp2/device_tuning.h`](../../src/core/tp2/device_tuning.h))
is max(3 x observed, 8 MiB) per topology class and batch size, from the memory `prepare_graphs()`
consumed per rank on two RTX 5070 Ti at 32K context and concurrency 1: 2 MiB ordinary and MTP3
(one class each, 8 MiB), 18/12 MiB on rank 0/1 for DFlash2 K=4 (five classes, 11 MiB each). The
server logs observed against allowance per rank and warns on an overrun.

The collectives have two transports ([`allreduce.h`](../../include/ninfer/ops/allreduce.h)). The
staged transport serves every eager call and every payload wider than a mailbox slot: each rank
pulls the peer's operand with a stream-ordered `cudaMemcpyAsync`, which the driver stages through
host memory without peer access, ordered by the Program's `PeerEvents`, then combines locally;
in a graph the events become edges. With CUDA Graphs the Program also owns a
`PeerMailbox` ([`peer_mailbox.h`](../../include/ninfer/ops/peer_mailbox.h))
attached to its `PeerEvents`, unless `EngineOptions::tp_mailbox` is false. An `allreduce_sum`
captured with both ranks' streams in one capture and a payload within a slot (`hidden x (K+1)`
BF16: every single-request decode, MTP-head and verification all-reduce) becomes one exchange
kernel per device that publishes its partial to pinned host memory, releases a per-slot epoch
flag, waits for the peer's and combines with the staged path's arithmetic, so both transports are
bit-identical. Epoch flags need no host reset between launches, and two alternating slots cover
any capture (see [`peer_exchange.cuh`](../../src/ops/kernel/peer_exchange.cuh)), so the pinned
slab is `2 x 2 x slot` bytes (160 KiB for MTP3 at hidden 5120). The mailbox is declared before
`peer_events` and the graph families and so outlives the kernels that address it. A poller that
waits about 0.8 s sets a sticky hang word; `synchronize_devices()` checks it after every retired
round and throws, since the ranks' results diverged.

## Vision and multimodal positions

The current native processor uses 16×16 spatial patches, pairs of frames, and 2×2 spatial merge.
It consumes bytes already acquired by the CLI or server, decodes images/video frames, resizes to
merge-aligned dimensions, normalizes RGB and packs BF16 patch rows. Each row contains
`3 * 2 * 16 * 16 = 1536` values in the defined channel/temporal/spatial order. Images repeat their
frame to supply the temporal pair.

The official Vision towers have depth 27, hidden width 1152, intermediate width 4304 and 16 heads
of dimension 72. The learned position table is a 48×48 grid. Patch projection and interpolated
position addition precede the transformer stack. Each block performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, two_dimensional_patch_positions)
a = segmented_attention(q, k, v)
x = x + output_projection(a) + bias
h = LayerNorm(x)
x = x + fc2(GELU_tanh(fc1(h) + bias)) + bias
```

Attention segments separate media/frame grids. The merger normalizes, groups 2×2 patches into
width 4608, and computes `fc2(GELU_exact(fc1(merged) + bias)) + bias`, producing Text-width columns.
Those columns replace matching placeholder embeddings. Vision has no autoregressive state.

Frontend admission uses aggregate media bytes, pixels, patch/token budgets and Engine context
capacity. The aggregate limit is 131,072 raw patches / 32,768 merged tokens; a single tower item
uses at most 16,384 merged tokens, additionally bounded by context. Program executes items
sequentially and reuses their output handoff after scattering the previous item.

The prepared prompt stores axis-major `[3,T]` positions in temporal, height, width order:

- Text tokens advance all three axes together.
- Media placeholders use their merged grid coordinates; following text resumes after the maximum
  multimodal coordinate.
- Video timestamps are rendered into prompt text. Position construction uses the prepared media
  grids and token runs.
- `rope_delta = next_multimodal_position - token_count` supplies the offset for subsequent decode.

For the official Text geometry, 64 of each 256-dimensional Q/K head rotate, using 32 frequency
pairs assigned to interleaved sections `[11,11,10]`. Text-only positions reduce to ordinary partial
RoPE because the three axes are equal.

## Vocabulary and frontend resources

Keep three row domains distinct: config `vocab_size` is the embedding/output matrix extent; the
tokenizer defines addressable token IDs; an optional proposal map defines its own shortlist rows.
The official tokenizer has 248,077 addressable IDs within 248,320 matrix rows. Remaining matrix
rows are not assumed zero and do not become public tokens.

Token spelling, special-token roles and EOS are derived from loaded resources. The frontend checks
the tokenizer's supported BPE/normalization semantics and token-domain agreement. Template bytes
select a compiled Qwen template; arbitrary Jinja execution is not implemented. Mode sampling
presets are architecture-owned, with explicit application/request overrides. Resource loading
does not infer execution identity from tokenizer filenames, release names or sampling values.

## State and numerical boundaries

Each StateImage contains the Text GDN state and continuation hidden. Per GDN layer, recurrent
state has one `[Dv,Dk]` matrix per value head, and convolution history has
`(2 * key_width + value_width) * (kernel_width - 1)` values. Program sizes Device/Host StateImage
capacity under the [context resource contract](resource-scheduling-and-context-cache.md#11-配置语义与有界性).
ReplaySSM records are separate pending-round scratch when speculation is enabled.

Text GQA KV grows with visible context. Selected MTP owns separate KV; DFlash2 owns the state described
in its reference. Continuation hidden values and prefix checkpoints belong to
Program. Workspace, storage capacities and graph buffers are derived from actual parameters and
startup options by the planners described in [Engine architecture](engine-architecture.md).

Speculation verifies the candidate sequence against processed target distributions, commits the
accepted prefix and continuation state, then publishes output. Greedy acceptance compares a
proposal token with the target argmax for that verify column. Sampling uses the defined proposal
acceptance and correction distribution. This does not impose token or logits equality between
different quantization, prefill or kernel paths.

Each floating-point Op is qualified against its independent mathematical oracle over represented
public inputs. Packed weights are decoded with their stored scales. Exact transforms use exact
oracles; private accumulation, staging and reduction order use the Op's numerical criteria.
[Op development](op-development.md) owns that qualification contract.
