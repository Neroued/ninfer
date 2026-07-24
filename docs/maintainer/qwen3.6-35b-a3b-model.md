# Qwen3.6-35B-A3B Model Reference

This reference records the exact BF16 checkpoint's Text, sparse-MoE, MTP, Vision,
multimodal-position, numeric, and persistent-state semantics, together with the model-side
semantics of the optional Qwen3.6-35B-A3B-DFlash companion. It does not define artifact bytes,
weight loading, packing or quantization, execution-Op design, or qualify the advertised extended
million-token mode. The registered artifact and quantization specification is
[`qwen3.6-35b-a3b-artifact.md`](qwen3.6-35b-a3b-artifact.md).

## 1. Model identity

Qwen3.6-35B-A3B is the first open-weight Qwen3.6 checkpoint. Qwen announced and open-sourced it on
2026-04-15, with weights distributed through Hugging Face and ModelScope. It is a post-trained
native multimodal sparse-MoE model with three checkpoint components:

- a 40-layer hybrid Text decoder with a sparse MoE in every layer;
- a one-layer MTP draft model whose decoder block also contains sparse MoE;
- a 27-layer Vision transformer and patch merger.

The checkpoint uses the Hugging Face architecture identifiers
`Qwen3_5MoeForConditionalGeneration`, `qwen3_5_moe`, and `qwen3_5_moe_text`. Those identifiers name
the implementation family; they do not make this checkpoint Qwen3.5 or indicate a local renaming.

`z-lab/Qwen3.6-35B-A3B-DFlash` is a separately distributed, optional BF16 draft companion for this
exact target. It is not a standalone language model, a fourth component of the base checkpoint, or
a replacement for the built-in MTP module. Pairing it with the target adds a proposal mechanism;
the target remains the authority for every published token.

The fixed shape and tensor inventory below are checkpoint facts. A different hidden size, layer
count, expert layout, head layout, or Vision tower is a different exact model profile rather than a
runtime configuration of this one.

## 2. Global dimensions

### 2.1 Text decoder

| Field | Value |
|---|---:|
| hidden size | 2048 |
| decoder layers | 40 |
| vocabulary matrix rows | 248320 |
| tokenizer-addressable tokens | 248077 |
| full-attention interval | 4 |
| full-attention layers | 10 |
| Gated-DeltaNet layers | 30 |
| routed experts per layer | 256 |
| selected routed experts per token | 8 |
| shared experts per layer | 1 |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1e7` |
| checkpoint position capacity | 262144 |
| MTP hidden layers | 1 |

Layer `i` is full attention exactly when `(i + 1) % 4 == 0`, giving zero-based indices
`3, 7, ..., 39`; every other layer is GDN. All 40 layers apply the same sparse-MoE structure after
their token mixer. There are no dense-FFN-only Text layers.

The input embedding and output head are untied, independent matrices of shape `[248320,2048]`.
The 248320 rows are the padded model vocabulary width. Tokenizer ids end at 248076, so rows
248077 through 248319 are padded/reserved model rows without corresponding tokenizer tokens; they
are not zero-valued rows.

The source config has no RoPE-scaling block and directly defines 262144 positions. The model card
separately documents an optional static YaRN factor of four for up to 1,010,000 tokens and warns
that enabling it can affect shorter inputs. That extended mode is a deployment transformation, not
the checkpoint's native position configuration.

### 2.2 Full gated attention

| Field | Value |
|---|---:|
| query heads | 16 |
| KV heads | 2 |
| head dimension | 256 |
| Q width | 4096 |
| output-gate width | 4096 |
| raw `q_proj` rows | 8192 |
| K/V width | 512 each |
| rotated dimensions per head | 64 |
| Q heads per KV head | 8 |
| attention scale | `1/sqrt(256) = 0.0625` |

The query projection is logically a per-head `[query_256 | output_gate_256]` projection. Query and
gate values are interleaved by head in the physical `[8192,2048]` checkpoint tensor; splitting the
first 4096 rows from the last 4096 rows is wrong. Q and K use zero-centered RMSNorm before partial
interleaved MRoPE. The causal-attention result is multiplied elementwise by `sigmoid(output_gate)`
before the output projection.

### 2.3 Gated-DeltaNet

| Field | Value |
|---|---:|
| Q/K heads | 16 |
| Q/K head dimension | 128 |
| V heads | 32 |
| V head dimension | 128 |
| Q width | 2048 |
| K width | 2048 |
| V width | 4096 |
| Z output-gate width | 4096 |
| fused Q/K/V projection width | 8192 |
| A/B control width | 32 each |
| causal convolution width | 4 |
| recurrent-state dtype | FP32 |
| delta-rule scale | `1/sqrt(128)` |

Every two adjacent V heads share one Q/K head. Each GDN layer retains three preceding Q/K/V
projection columns and 32 recurrent matrices of shape `[128,128]`; its persistent state is bounded
with respect to context length.

Verify `T=1..6` invokes the shared `gdn_input_proj_conv_snapshot` contract. The exact W8 leaf feeds
the projection accumulators for the first 8192 rows directly into causal convolution and SiLU,
writes q/k/v and BF16 snapshots to slots `0..T-1`, and writes the final 4096 Z rows directly. The
former materialized qkv tensor is not a semantic cast boundary.

### 2.4 Sparse MoE

| Field | Value |
|---|---:|
| logical routed experts | 256 |
| selected routed experts | 8 per token |
| routed-expert intermediate size | 512 |
| shared-expert intermediate size | 512 |
| router output width | 256 |
| selected-weight normalization | sum to 1 after top-k |
| shared-expert gate | one sigmoid scalar per token |

Every routed expert and the shared expert is a 512-wide SwiGLU. The shared expert is not expert 256,
does not participate in top-k, and always executes. There is no inference capacity factor, token
dropping, stochastic routing, or auxiliary-loss term in the forward pass.

### 2.5 Vision

| Field | Value |
|---|---:|
| transformer depth | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| head dimension | 72 |
| spatial patch | 16 × 16 |
| temporal patch | 2 frames |
| flattened patch width | `3 × 2 × 16 × 16 = 1536` |
| spatial merge | 2 × 2 patches |
| merger input width | 4608 |
| merger output width | 2048 |
| learned position table | 48 × 48 |
| Vision RoPE theta | 10000 |
| Vision LayerNorm epsilon | `1e-6` |
| deep-stack injection layers | none |

The Vision backbone geometry matches Qwen3.6-27B, but its merger is checkpoint-specific and emits
width 2048 rather than 5120. `deepstack_visual_indexes` is empty and the checkpoint contains no
deep-stack merger weights.

### 2.6 DFlash companion

The values in this section are from companion revision
`f181eece646affea2c38b2765f1aaa01a9734ccd`, the Z-Lab/Modal long-context retrain mirrored by the
local `dflash-bf16` directory. Earlier published DFlash revisions with different layer counts,
feature-layer ids, or mask ids are different draft profiles.

| Field | Value |
|---|---:|
| draft hidden layers | 6 |
| hidden size | 2048 |
| dense SwiGLU intermediate size | 6144 |
| query heads | 32 |
| KV heads | 8 |
| head dimension | 128 |
| Q width | 4096 |
| K/V width | 1024 each |
| Q heads per KV head | 4 |
| attention scale | `1/sqrt(128)` |
| layer attention | 5 non-causal sliding + 1 non-causal full |
| sliding-window parameter | 4096 |
| position capacity | 262144 |
| RMSNorm epsilon | `1e-6` |
| one-dimensional RoPE theta | `1e7` |
| vocabulary rows through shared target head | 248320 |
| native total block length | 16 |
| parallel mask proposals per native block | 15 |
| internal mask row | 248077 |
| target feature layers, zero-based | `1, 6, 11, 16, 22, 27, 32, 37` |
| private BF16 tensors | 69 |
| private parameters | 385,906,176 |

The companion is six dense Qwen3 decoder blocks, not a copy of the target's hybrid GDN/MoE
topology. It contains no embedding table, output head, Vision tower, GDN state, MoE, attention
output gate, or projection bias. It reuses the target input embedding and independent target
`lm_head`. Its internal mask id 248077 is the first reserved model row after the
tokenizer-addressable range, not an encodable text token.

The private parameter count comprises a bias-free `[2048,16384]` target-feature projection,
dedicated hidden/final norms, and the six draft blocks. It is separate from the base-checkpoint
inventory below.

### 2.7 Exact base-checkpoint parameter inventory

Header-only inspection of all 26 safetensors shards gives 1045 BF16 tensors and the following exact
weight-element counts:

| Component | Parameters |
|---|---:|
| token embedding | 508,559,360 |
| 40 Text decoder blocks | 33,643,489,920 |
| final Text norm and independent `lm_head` | 508,561,408 |
| one-layer MTP module | 844,640,768 |
| Vision encoder and merger | 446,571,248 |
| **total checkpoint** | **35,951,822,704** |

The official “35B total / 3B activated” description is rounded and uses an active-parameter
convention, not a residency claim. All 256 expert banks must remain available because the next token
may route to any expert. If all main-Text parameters except `lm_head` are counted while the 256
routed banks are replaced by the eight selected banks, the local inventory gives 2,946,429,568
parameters, which rounds to A3B. Evaluating the independent full `lm_head` accesses another
508,559,360 weights; parameter counts are not FLOP or memory-traffic counts.

## 3. Shared decoder layer skeleton and norms

Let zero-centered RMSNorm and plain RMSNorm be:

```text
offset_rmsnorm(x,w) = (1 + w) * x / sqrt(mean(x^2) + eps)
plain_rmsnorm(x,w)  =       w  * x / sqrt(mean(x^2) + eps)
```

All base-checkpoint Text/MTP input, post-attention, and final norms, the MTP stem norms, and
full-attention Q/K norms use `offset_rmsnorm`. The GDN internal output norm is the only Text/MTP
plain RMSNorm. The external DFlash companion instead uses `plain_rmsnorm` throughout. Vision
uses ordinary LayerNorm with learned weight and bias.

Both Text layer types use the same pre-norm residual schedule:

```text
h = offset_rmsnorm(x, input_norm)
x = x + mixer(h)                         # GDN or gated full attention
h = offset_rmsnorm(x, post_attention_norm)
x = x + sparse_moe(h)                    # top-8 routed + gated shared expert
```

Text and MTP projections, expert matrices, routers, and GDN convolution have no bias. Attention
dropout is zero. Fused projections, delayed residual addition, expert parallelism, and combined
kernel launches may change the execution schedule but not these mathematical boundaries.

## 4. Full-attention layer

For normalized input `h[2048,T]`:

```text
q_gate = q_projection(h)                  # [16,512,T]
q, gate = split_last(q_gate, [256,256])   # [16,256,T] each
k = k_projection(h)                       # [2,256,T]
v = v_projection(h)                       # [2,256,T]

q = offset_rmsnorm(q, q_norm)
k = offset_rmsnorm(k, k_norm)
q, k = partial_interleaved_mrope(q, k, rotary_dims=64)

a = causal_gqa(q, k, v, scale=1/sqrt(256), kv_cache)
a = a * sigmoid(gate)
y = o_projection(a)                       # [2048,T]
```

Each KV head serves eight query heads. Prefill appends all K/V columns and evaluates causal
attention over the chunk. Decode appends one column and attends over the resident prefix. A cache
backend may page or quantize K/V, but those are representation choices rather than model math.
The persistent cache boundary is the offset-normalized, MRoPE-rotated K and the directly projected
V; raw K, Q, and the output gate are not cached.

Only 64 of each 256-dimensional Q/K head are rotated. They contain 32 complex frequency pairs split
across temporal, height, and width MRoPE sections `[11,11,10]`. The checkpoint sets
`mrope_interleaved=true`, so axis selection is interleaved by frequency pair. For text-only input,
all three position rows are equal and the result is exactly ordinary one-dimensional partial RoPE.

The gate half of `q_proj` is neither normalized nor rotated. It bypasses attention and is applied
only after the per-head value aggregation, immediately before `o_proj`.

## 5. Gated-DeltaNet layer

The normalized input first produces logical Q/K/V, the output gate Z, and per-V-head controls:

```text
qkv = in_qkv(h)    # [8192,T] = q[16,128,T] | k[16,128,T] | v[32,128,T]
z   = in_z(h)      # [32,128,T]
a   = in_a(h)      # [32,T]
b   = in_b(h)      # [32,T]
```

Only concatenated Q/K/V passes through a depthwise causal width-4 convolution followed by SiLU.
Z, A, and B do not pass through the convolution. The convolved Q and K are L2-normalized per head
with epsilon `1e-6`; GDN consumes no position ids and applies no RoPE. The production GDN Op always
receives raw BF16 q/k. Its recurrent implementation retains normalized values in FP32 registers;
when it selects the chunked implementation, it privately materializes normalized BF16 q/k for the
chunked body and recurrent tail.

The decay and update controls `g` and `beta` are observable FP32 values. Their logical formula for
every V head is:

```text
g     = -exp(A_log) * softplus(a + dt_bias)
alpha = exp(g)
beta  = sigmoid(b)
```

For V head `j`, Q and K come from head `j // 2`. Using the document's state convention
`S[128_v,128_k]`, one token performs:

```text
k     = k / sqrt(sum(k^2) + 1e-6)
q     = q / sqrt(sum(q^2) + 1e-6)
Sbar  = alpha * S
Sk    = Sbar @ k
u     = beta * (v - Sk)
S     = Sbar + u outer k
o     = (S @ q) * (1/sqrt(128))
```

This ordering matters: the delta error uses the already-decayed `Sbar`. Kernel implementations may
transpose the physical state or combine these operations, but must remain algebraically equivalent.

The per-head result is normalized and gated before projection:

```text
on = plain_rmsnorm(o, gdn_norm) * SiLU(z)  # [32,128,T]
y  = out_projection(on)                    # [2048,T]
```

Each GDN layer owns three previous convolution columns and one FP32 `[32,128,128]` recurrent-state
set for ordinary continuation. Prefill may use a parallel chunked delta-rule algorithm and decode a
recurrent algorithm. Speculative decoding or prefix reuse requires extra state slots, but cannot
change the recurrence licensed by a committed prefix.

## 6. Sparse-MoE layer

For post-mixer normalized token state `h[2048]`, the router and selected weights are:

```text
router_logits = W_router h                         # [256], no bias
router_probs  = softmax(router_logits)
(values, ids) = topk(router_probs, 8)
routing_weight = values / sum(values)
```

For each selected routed expert `e`, the source tensor stores gate rows before up rows:

```text
gate, up = split(W_gate_up[e] h, [512,512])
y_e = W_down[e](SiLU(gate) * up)                   # [2048]
y_routed = sum(routing_weight[e] * y_e for e in ids)
```

The independent shared path is:

```text
shared = W_shared_down(
    SiLU(W_shared_gate h) * W_shared_up h
)
shared_scale = sigmoid(W_shared_score h)           # scalar

sparse_moe(h) = y_routed + shared_scale * shared
```

Softmax is taken over all 256 router logits before top-k, and the selected values are then
renormalized to sum to one. This is algebraically equivalent to a softmax over only the selected
logits. Hugging Face casts selected weights to activation dtype, while vLLM keeps them FP32 through
its fused expert path. Those are execution profiles, not alternative mathematical oracles, and do
not define a cross-path equality requirement. Router auxiliary loss coefficient `0.001` and
`output_router_logits=false` are training/output configuration; neither adds an inference term.

Expert grouping, token permutation, physical expert reordering, shared-expert fusion, and separate
decode/prefill kernels are execution policies. Logical expert id `e` remains router row `e`, all
assignments are retained, and the shared branch is always added.

## 7. Text prefill and decode

### Prefill

1. gather token embeddings into `[2048,T]` without an embedding scale;
2. replace image/video placeholder columns with Vision merger output when input is multimodal;
3. run the 40 decoder layers in `[GDN, GDN, GDN, full attention] × 10` order, applying sparse MoE
   after every mixer;
4. carry full-attention KV and GDN convolution/recurrent state across chunks without resetting
   logical positions;
5. apply final offset RMSNorm;
6. project required hidden columns through the independent `lm_head[248320,2048]`;
7. process logits and select the next token according to the generation policy.

Chunking is an implementation workspace policy. It must not restart the causal convolution, GDN
matrix state, KV cache, or MRoPE positions.

### Ordinary decode

1. embed the current token;
2. run all 40 layers for one new position;
3. append ten K/V entries and update thirty GDN state sets;
4. apply final norm and the full `lm_head`;
5. process logits, select the next token, and commit the new logical position.

The full target head remains the correctness boundary. A shortlist, vocabulary partition, or fused
top-k head is valid only if it is proven equivalent to the required output policy.

## 8. MTP draft model

The checkpoint contains one MTP decoder layer. It is conditioned on both the target Text hidden
state and the token following that state; it is not an auxiliary `lm_head` directly attached to the
main decoder.

For main-model final-normalized hidden state `h_t` and token `x_(t+1)`:

```text
e = offset_rmsnorm(embed(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t,             pre_fc_norm_hidden)
u = fc(concat(e, h))                                      # [4096] -> [2048]

u = one_full_attention_plus_moe_decoder_layer(u)
draft_hidden = offset_rmsnorm(u, mtp.norm)
draft_logits = shared_target_lm_head(draft_hidden)         # predicts x_(t+2)
```

The MTP decoder layer has its own full gated-attention and MoE weights with the same geometry as a
Text full-attention layer: 16 Q heads, 2 KV heads, 256 head dimension, gated attention output,
256 routed experts selecting eight, and one gated shared expert. It has its own one-layer KV cache.
Its Q/K use offset RMSNorm and partial interleaved `[11,11,10]` MRoPE over the first 64 dimensions;
its per-head attention output uses the same sigmoid gate, and its residual order is exactly the
schedule in Section 3.

`mtp_use_dedicated_embeddings=false`, and the checkpoint contains neither an MTP embedding nor an
MTP head. Inference therefore reuses the target embedding and independent full `lm_head`. The MTP
final norm is dedicated. Ordinary Hugging Face target-model loading may ignore `mtp.*` weights;
using the draft module requires an inference path that loads and schedules them explicitly.

### Sequence and prefill alignment

For a target sequence `x_0, ..., x_n` with final-normalized target hidden states
`h_0, ..., h_n`, after the target selects `x_(n+1)`, MTP prefill uses:

```text
MTP token inputs    = [x_1, x_2, ..., x_n, x_(n+1)]
MTP hidden inputs   = [h_0, h_1, ..., h_(n-1), h_n]
MTP position inputs = [p_0, p_1, ..., p_(n-1), p_n]
```

Thus each MTP position `p_t` fuses `h_t` with `embed(x_(t+1))`; token ids are shifted left, while
target hidden states and positions are not. The complete aligned sequence passes causally through
the MTP layer to populate its independent KV cache, and the final prefill logit proposes
`x_(n+2)`. Starting the MTP layer only at the last prompt token with an empty cache, or assigning
the shifted token position `p_(t+1)`, is incorrect.

Here `embed(x_(t+1))` means the same composed input-embedding semantics as target Text. When the
shifted position falls inside an image/video placeholder span, the corresponding Vision merger
column replaces the placeholder-token row before the MTP stem. Reusing the target embedding table
does not permit MTP to discard multimodal embedding replacement.

One MTP hidden layer does not mean one possible draft token. After the first proposal, an inference
engine may feed the proposed token and previous MTP hidden state back through the same MTP module at
the next position, advancing its KV state to propose further tokens.

## 9. DFlash block-diffusion draft model

DFlash replaces repeated autoregressive draft steps with one masked-block forward. It is called a
block-diffusion drafter because the draft model denoises several masked positions jointly, but
inference performs neither an iterative diffusion schedule nor a candidate tree. One forward
produces the entire ordered proposal block, and the target then verifies that block causally.

In the DFlash training objective, one clean anchor precedes a block whose remaining positions are
replaced by the internal mask row. Frozen target residual features condition the drafter, and the
original tokens at all masked positions supply cross-entropy targets in one pass; position-dependent
loss weighting emphasizes the earlier proposals whose errors would truncate more of the accepted
prefix. The target model and its shared embedding/output head remain fixed. A deployment may use
either that full output head or the artifact's optimized proposal shortlist to select DFlash
drafts; this does not alter the trained DFlash layers. This objective explains the block-parallel
inference geometry but does not add a diffusion timestep or iterative sampler.

### Target conditioning

Let `r_t^l` be the complete residual-stream output after zero-based target Text block `l` at
position `t`, before the next block's input norm or the final Text norm. The companion selects eight
such outputs and computes:

```text
s_t = concat(r_t^1, r_t^6, r_t^11, r_t^16,
             r_t^22, r_t^27, r_t^32, r_t^37)       # [16384]
c_t = plain_rmsnorm(W_fc s_t, hidden_norm)           # [16384] -> [2048]
```

`W_fc` is bias-free. The selected ids refer to target decoder blocks, not hidden-state-array
indices; an API that includes the embedding state at index zero may therefore expose these values
at boundaries `2, 7, 12, 17, 23, 28, 33, 38`.

The same `c_t` is direct context for all six draft layers. It is not added to the draft residual
stream and does not pass through the draft Q projection, output projection, or MLP. Instead, each
layer independently projects every committed context position into that layer's K/V space:

```text
K_ctx^l(t) = rope_1d(
    plain_head_rmsnorm(W_k^l c_t, k_norm^l), position=t
)
V_ctx^l(t) = W_v^l c_t
```

Thus the persistent DFlash context represents target-produced features, not draft hidden states
recursively advanced through six layers. Initial prefill applies the feature projection and six
K/V projections to each target-prefix position; it does not run the prefix through the six draft
residual/MLP stacks. Decode extends this context only from newly committed target residual outputs.

### Parallel proposal block

Suppose the target has processed through token `x_t` and selected the still-unprocessed anchor
`a = x_(t+1)`. For native total block length `B=16`, the DFlash query is:

```text
absolute position  t+1       t+2       ... t+16
input row          a         MASK      ... MASK
role               anchor    proposal      proposal
```

The initial query residuals are the target embeddings of these 16 rows. The anchor row conditions
the block but is not sampled. After one six-layer forward, final plain RMSNorm is applied and only
the 15 mask columns produce proposals. The full route applies the shared target
`lm_head [248320,2048]` and takes argmax over rows `0..248076`. The optimized route applies
`text/draft_head [131072,2048]`, takes shortlist argmax, and remaps the selected row through
`text/draft_head_token_ids`. Either route produces a valid speculative proposal; shortlist output
need not equal full-head argmax because causal target verification remains the output correctness
boundary. A mask row predicts the token at that same absolute position; there is no
causal-language-model one-position shift. Consequently, the checkpoint's `block_size=16` means one
anchor plus 15 proposals. In an API where `num_speculative_tokens` counts proposals, the matching
value is 15 rather than 16.

The published companion also permits a shorter total block, such as eight rows. A total block of
`B` rows always contains one anchor and `B-1` masks. Because the final layer is non-causal, changing
`B` can change every proposal logit rather than merely truncate a fixed 16-row result.

All proposals are obtained in this one forward. Sampled candidates are not fed back through the
draft model before target verification, and there is no iterative mask refinement.

### Draft-layer computation

For draft-layer input `u_l`, let `n_l = plain_rmsnorm(u_l, input_norm_l)`. Each layer computes its
query-block K/V in addition to the context K/V above:

```text
Q_l       = rope_1d(plain_head_rmsnorm(W_q^l n_l, q_norm_l), query_positions)
K_query_l = rope_1d(plain_head_rmsnorm(W_k^l n_l, k_norm_l), query_positions)
V_query_l = W_v^l n_l

A_l = attention(
    Q_l,
    concat_sequence(K_ctx_l, K_query_l),
    concat_sequence(V_ctx_l, V_query_l),
    scale=1/sqrt(128),
    layer_mask_l
)

v_l = u_l + W_o^l A_l
m_l = plain_rmsnorm(v_l, post_attention_norm_l)
u_(l+1) = v_l + W_down^l(SiLU(W_gate^l m_l) * W_up^l m_l)
```

Q has 32 heads, K/V have eight heads, and four Q heads share each K/V head. Q/K head norm and RoPE
cover all 128 head dimensions. This is ordinary one-dimensional split-half Qwen3 RoPE with
`theta=1e7`, not the target's partial three-axis MRoPE. All companion norms are multiplicative
`plain_rmsnorm`; none use the target's offset-norm convention. `plain_head_rmsnorm` is the same
formula applied independently over each 128-value Q or K head.

All six DFlash layers are non-causal. Layers 0 through 4 apply symmetric non-causal local attention
with the configured sliding-window parameter 4096, and layer 5 applies non-causal full attention.
For committed context length `L`, query-block length `B`, query position `p_i=L+i`, and any
populated context or query key position `p_j`, the local-layer predicate is:

```text
allowed_swa(p_i, p_j) = abs(p_j - p_i) < 4096
                       = p_i-4095 <= p_j <= p_i+4095
```

The endpoints at distance 4095 are inclusive and positions at distance 4096 are excluded. This is
a symmetric mask whose maximum populated extent is 8191 positions, not a 4096-key total span.
Context keys occupy positions `0..L-1` and temporary query keys occupy `L..L+B-1`; the valid
sequence bounds clip the interval above. Because `B<=16`, every query row attends all temporary
query K/V rows. Once the left boundary is populated, row `i` attends context positions
`L+i-4095..L-1`, or `4095-i` context keys.

This exact inequality follows the checkpoint reference's FlashAttention path: Transformers maps
the scalar value 4096 to the inclusive FlashAttention window `(4095,4095)`, and FlashAttention
bottom-right-aligns the shorter Q block with the complete K sequence. The reference's unmasked
SDPA fallback becomes full attention beyond this boundary and is not an alternate model semantic
or numerical oracle.

The query rows therefore attend bidirectionally to one another while also attending the admitted
context K/V. This all-non-causal mask is the single model contract and numerical oracle; a causal
draft mask computes a different head and is not a supported execution profile.

The companion consumes target Text residual features and owns no Vision component. Its query and
context positions are scalar one-dimensional positions. This model-side contract does not by
itself claim a product execution route for image/video prompts.

## 10. Speculative verification and state transaction

MTP and DFlash are proposal mechanisms. The main 40-layer target remains the only output authority.
MTP advances its one-layer drafter autoregressively for each candidate, whereas DFlash proposes all
15 mask positions in parallel; this scheduling difference does not change target verification.

For DFlash, the target consumes the anchor followed by the 15 candidates in one causal forward.
Unlike the same-position DFlash mask logits, target logits retain ordinary next-token alignment:

```text
target input row       anchor at t+1    ... candidate at t+15    candidate at t+16
target logit checks    candidate t+2    ... candidate t+16       emits bonus t+17
```

Greedy verification scans left to right and accepts the longest candidate prefix equal to the
target argmax at each corresponding position. At the first mismatch it publishes the target token
for that position; if all candidates match, it publishes the target bonus following the block. If
`k` proposals are accepted, that verification therefore publishes `k+1` tokens beyond the anchor,
with a maximum of 16 for the native block. Any reported mean acceptance length must state whether
this mandatory correction/bonus token is included.

Sampling verification must use proposal and target probabilities with an acceptance/correction
rule that preserves the processed target distribution; equality-only checking is sufficient only
for greedy decoding. Draft quality changes acceptance and throughput, never the target-model
distribution.

Only the target state through the anchor and accepted candidate prefix becomes committed. The
correction or bonus is the next still-unprocessed anchor. Rejected target state and draft-query K/V
must not remain logically reachable. Continuation is defined by the accepted-prefix snapshots of
all ten target KV caches, thirty GDN convolution windows, and thirty recurrent-state sets.

For DFlash specifically, temporary K/V produced from anchor/mask query hidden states is speculative
branch state, not durable context. After verification, target residual outputs for the newly
committed positions are transformed through `W_fc` and the six per-layer K/V projections to extend
the DFlash context. Rejected-tail features are discarded. Stale bytes may remain only in
unreferenced physical storage.

## 11. Vision preprocessing

The checkpoint processor accepts image and video media. For each media item it:

1. decodes the image or samples video frames;
2. chooses spatial dimensions aligned to the 32-pixel `patch_size × spatial_merge_size` factor;
3. resizes RGB input and normalizes each channel as `(pixel - 0.5) / 0.5`;
4. groups two frames and packs channel-major `2 × 16 × 16` patches into FP32 rows of width 1536;
5. expands one image frame into the required temporal pair, or groups video frames in pairs;
6. records each media grid as `(grid_t, grid_h, grid_w)` before spatial merge;
7. emits one Text placeholder per merged 2×2 patch group and records its scatter span;
8. constructs timestamps, token types, and three-axis Text positions for later MRoPE.

The source processor config uses image pixel-count bounds 65,536 through 16,777,216 and video
pixel-frame bounds 4,096 through 25,165,824. These are frontend work budgets rather than learned
model dimensions. Before other limits, the resulting Vision-token count is
`grid_t × grid_h × grid_w / 4`.

## 12. Vision tower

Patch rows are converted to the model activation dtype and projected with a biased Conv3D kernel
`[1152,3,2,16,16]`. The runtime adds a learned 2304-entry position table interpreted as 48×48 and
bilinearly interpolated to each spatial grid. That learned table is repeated over temporal grids.

Each patch head has dimension 72. Vision RoPE uses the complete head, with height and width positions
providing the two 36-dimensional sections. For every temporal grid slice, each of the 27 blocks
performs non-causal self-attention and an MLP:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_2d_rope(q, k, height_width_positions, theta=10000)
a = segmented_attention(q, k, v, scale=1/sqrt(72), cu_seqlens)
x = x + projection(a) + bias

h = LayerNorm(x)
h = GELU_tanh(fc1(h) + bias)
x = x + fc2(h) + bias
```

Cumulative sequence lengths segment attention by temporal grid slice. Patches within one slice
attend bidirectionally to one another; unrelated images, videos, and temporal slices do not attend
through the Vision transformer. Temporal information within each pair has already entered through
the temporal patch projection.

The final merger applies LayerNorm to each 1152-wide patch token, groups every spatial 2×2 block
into width 4608, and computes:

```text
visual = fc2(GELU_exact(fc1(merged) + bias)) + bias    # [2048,V]
```

The merger's GELU is the exact erf form, unlike the tanh-approximated GELU in Vision-block MLPs.
The resulting columns replace the matching Text placeholder embeddings. This checkpoint has no
deep-stack feature injection. Autoregressive Text decode no longer consumes Vision activations
after this replacement, so a runtime may release them after prefill.

## 13. Multimodal Text positions

Text attention consumes axis-major positions `[3,T]` in temporal, height, width order.

- ordinary text tokens advance all three axes together;
- image/video placeholder tokens use positions derived from their merged Vision grid and, for
  video, temporal metadata;
- text following a media span resumes after the maximum multimodal position;
- `rope_delta = next_multimodal_position - token_count` converts later scalar decode indices into
  the continuing MRoPE position.

Full-attention Q/K consumes these positions through the interleaved `[11,11,10]` MRoPE sections.
GDN does not consume positions. After multimodal prefill, Vision can be released while Text decode
continues with the saved `rope_delta`, KV caches, and GDN state.

## 14. Vocabulary and generation boundaries

The tokenizer's base BPE vocabulary has 248044 entries and the local tokenizer resolves 33 added
tokens, giving 248077 addressable ids. Important checkpoint ids are:

| Meaning | Token | Id |
|---|---|---:|
| padding / end of text | `<|endoftext|>` | 248044 |
| chat message start | `<|im_start|>` | 248045 |
| chat message end | `<|im_end|>` | 248046 |
| vision start | `<|vision_start|>` | 248053 |
| vision end | `<|vision_end|>` | 248054 |
| image placeholder | `<|image_pad|>` | 248056 |
| video placeholder | `<|video_pad|>` | 248057 |
| thinking start/end | `<think>`, `</think>` | 248068, 248069 |

The tokenizer declares EOS 248046 and padding 248044. `generation_config.json` stops on either
248046 or 248044 and uses 248044 for BOS/padding. The Text config's BOS/EOS fields both contain
248044, so a frontend must not collapse configuration-level and tokenizer/generation-level
meanings into one inferred id.

The tokenizer also defines reserved audio-related tokens, but the checkpoint contains no audio
encoder or audio projection tower. Token presence is not evidence of an audio input modality.

## 15. Precision and reference boundaries

- All 1045 base-checkpoint tensors are BF16; the base source has no quantization configuration.
- All 69 private DFlash-companion tensors are BF16. Its target embedding is the target's existing
  matrix. Proposal selection may use the existing full `lm_head` or the artifact's existing
  optimized proposal head; neither is a private companion parameter.
- Public activation, cache, and recurrent-state dtypes are stated by their owning Op/state contract.
  In particular, GDN recurrent matrices and decay controls are FP32, while registered BF16/INT8 KV
  formats remain real persistent representation boundaries.
- Every floating-point Op uses one independent naive FP32/FP64 mathematical oracle over its logical
  inputs. Packed weights are decoded from their stored codes and exact stored scales. Exact
  transforms and codecs use exact oracles.
- Hugging Face, vLLM, llama.cpp, and the artifact-native Python route are execution/parity profiles,
  not Op oracles. Their choices to cast attention probabilities, GDN controls, gated-norm values,
  MoE route weights, expert activations, or convolution history do not bind NInfer kernels.
- NInfer kernels may select their natural accumulator precision, operand staging, intermediate
  materialization, workspace dtype, and reduction association. Each route is accepted directly
  against the one Op oracle with its own documented tolerance; the oracle precision does not become
  a required kernel operation order.
- The independent full `lm_head` remains part of target correctness even when a draft or fused
  sampling path avoids materializing all logits. Weight quantization, KV quantization, expert
  reordering, and packed layouts are derived runtime representations rather than source-model
  mathematics.
- DFlash proposal parity uses the all-non-causal mask defined in Section 9. A causal draft mask
  changes the logical formula and cannot serve as a numerical or behavioral oracle.

## 16. Persistent state inventory

The logical per-sequence state is:

| State | Shape basis | BF16/FP32 payload at 262144 context | Lifetime |
|---|---|---:|---|
| Text GQA K and V | 10 layers × context × 2 heads × 256 × 2 planes | 5.0 GiB BF16 | active sequence |
| MTP K and V | 1 layer × context × 2 heads × 256 × 2 planes | 0.5 GiB BF16 | active sequence when MTP enabled |
| DFlash sliding context K and V | 5 layers × non-causal local context × 8 heads × 128 × 2 planes | about 80 MiB BF16 | active sequence when DFlash enabled |
| DFlash full context K and V | 1 layer × context × 8 heads × 128 × 2 planes | 1.0 GiB BF16 | active sequence when DFlash enabled |
| GDN convolution history | 30 layers × 8192 channels × 3 columns | 1.406 MiB BF16 or 2.813 MiB FP32 | active sequence |
| GDN recurrent matrices | 30 layers × 32 heads × 128 × 128 | 60 MiB FP32 | active sequence |
| multimodal continuation | `rope_delta` and logical positions | negligible | active sequence |
| MoE route data | token × 8 ids and weights plus grouping/reduction workspace | implementation-dependent | operator scope |
| Vision workspace | patches, positions, QKV/MLP activations, merged embeddings | implementation-dependent | one multimodal prefill |

Allocator alignment, paging metadata, and scratch are not included. Speculative verification,
prefix reuse, or transactional rollback multiplies the bounded GDN state by the required snapshot
slots. Target full-attention KV, MTP KV, and the final DFlash layer's context KV grow with
configured context; GDN state and the first five DFlash context windows are bounded.
The first five layers may use 4096-slot cyclic K/V storage, but a proposal query admits at most the
last 4095 committed context positions according to its absolute position; a retained row at
distance 4096 is outside the mask. These caches do not grow with total context.

The Program freezes its feature set at startup. A disabled speculative backend has no proposal
model state or optimized proposal-head view. MTP and DFlash each load the optimized proposal head
only when that head route is selected; DFlash never loads MTP decoder weights or MTP KV state.
With Vision disabled, the Program has no Vision weight view and the shared workspace excludes the
maximum Vision envelope; media is rejected by the matching Frontend. The complete artifact
inventory is still validated before these resident views are published.

## 17. Base-checkpoint tensor layout

The following source shapes are part of the exact base-checkpoint identity. Linear weights use
`[out,in]` unless a batched expert or convolution shape is shown. DFlash loading and physical
tensor-layout contracts are intentionally outside this model reference.

### 17.1 Text roots

```text
model.language_model.embed_tokens.weight                 [248320,2048]
model.language_model.norm.weight                         [2048]
lm_head.weight                                            [248320,2048]
```

### 17.2 One GDN layer

```text
input_layernorm.weight                                    [2048]
linear_attn.in_proj_qkv.weight                            [8192,2048]
linear_attn.in_proj_z.weight                              [4096,2048]
linear_attn.in_proj_a.weight                              [32,2048]
linear_attn.in_proj_b.weight                              [32,2048]
linear_attn.conv1d.weight                                 [8192,1,4]
linear_attn.A_log                                         [32]
linear_attn.dt_bias                                       [32]
linear_attn.norm.weight                                   [128]
linear_attn.out_proj.weight                               [2048,4096]
post_attention_layernorm.weight                           [2048]
```

`in_proj_qkv` logical row order is contiguous Q, K, V. Physical runtime layouts may reorder V-head
groups for a fused recurrence only if the same permutation is applied to Z, A/B, decay parameters,
convolution channels, output-projection columns, and state.

### 17.3 One full-attention layer

```text
input_layernorm.weight                                    [2048]
self_attn.q_proj.weight                                   [8192,2048]
self_attn.k_proj.weight                                   [512,2048]
self_attn.v_proj.weight                                   [512,2048]
self_attn.q_norm.weight                                   [256]
self_attn.k_norm.weight                                   [256]
self_attn.o_proj.weight                                   [2048,4096]
post_attention_layernorm.weight                           [2048]
```

Within `q_proj`, each head contributes `[query_256 | gate_256]` before the next head.

### 17.4 MoE attached to every Text/MTP layer

```text
mlp.gate.weight                                           [256,2048]
mlp.experts.gate_up_proj                                  [256,1024,2048]
mlp.experts.down_proj                                     [256,2048,512]
mlp.shared_expert.gate_proj.weight                        [512,2048]
mlp.shared_expert.up_proj.weight                          [512,2048]
mlp.shared_expert.down_proj.weight                        [2048,512]
mlp.shared_expert_gate.weight                             [1,2048]
```

For each routed expert, rows `[0,512)` of `gate_up_proj` are the SiLU gate and rows `[512,1024)`
are the multiplicative up projection. Expert index is the leading tensor index and equals the
router-row identity.

### 17.5 MTP-only stem and final norm

```text
mtp.pre_fc_norm_embedding.weight                          [2048]
mtp.pre_fc_norm_hidden.weight                             [2048]
mtp.fc.weight                                              [2048,4096]
mtp.layers.0.*                                             full attention + MoE shapes above
mtp.norm.weight                                            [2048]
```

There is no `mtp.embed_tokens` or MTP-specific output head in this checkpoint.

### 17.6 Vision highlights

```text
model.visual.patch_embed.proj.weight                      [1152,3,2,16,16]
model.visual.patch_embed.proj.bias                        [1152]
model.visual.pos_embed.weight                             [2304,1152]
model.visual.blocks.*.attn.qkv.weight                     [3456,1152]
model.visual.blocks.*.mlp.linear_fc1.weight               [4304,1152]
model.visual.merger.linear_fc1.weight                     [4608,4608]
model.visual.merger.linear_fc2.weight                     [2048,4608]
```

Vision projections carry biases, and every Vision block contains two standard LayerNorm weight/bias
pairs. The tensor inventory contains no deep-stack merger and no audio tower.

## 18. Evidence and implementation map

This reference was cross-checked on 2026-07-23 against the following independent sources:

| Concern | Source |
|---|---|
| official identity, release, capabilities | [Qwen release blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b), [official repository](https://github.com/QwenLM/Qwen3.6) |
| official dimensions and context/YaRN statement | [Qwen Hugging Face model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| exact HF fields | [checkpoint `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) |
| shared implementation-family explanation | [Transformers Qwen3.5-MoE documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe) |
| exact local tensor shapes/counts | local `config.json`, `model.safetensors.index.json`, and safetensors headers |
| Text, GDN, MoE, MTP execution | vLLM commit `92221485aaaa4088491db3f182dd65a390fc9ac5`, especially `qwen3_5.py`, `qwen3_5_mtp.py`, `qwen3_next.py`, and `qwen_gdn_linear_attn.py` |
| independent Text/MTP recurrence graph | llama.cpp commit `07d937828636e305bc0cfe738b288f9ab05ff748`, especially `src/models/qwen35moe.cpp` and `src/models/delta-net-base.cpp` |
| Vision and processor semantics | vLLM `qwen3_vl.py` / `qwen2_5_vl.py`, llama.cpp `tools/mtmd/models/qwen3vl.cpp`, and local processor configs |
| DFlash algorithm, all-non-causal attention, block verification, and exact local-window boundary | [DFlash paper, revision 2](https://arxiv.org/html/2602.06036v2), [pinned Z-Lab PyTorch reference](https://github.com/z-lab/dflash/blob/94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756/dflash/model.py), [Transformers 5.7 FlashAttention mapping](https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/modeling_flash_attention_utils.py#L627-L632), [FlashAttention local-window definition](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#L253-L272) |
| exact DFlash companion identity and dimensions | [companion model card](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), [checkpoint `config.json`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/f181eece646affea2c38b2765f1aaa01a9734ccd/config.json), [companion PyTorch reference snapshot](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/0ce151d016ba8f0f0454d160a81b52ad4588c924/dflash.py), local safetensors header |

The registered implementation maps these concerns as follows:

| Model concern | Source |
|---|---|
| exact 2048-wide dimensions, 40-layer topology, limits, scales, and option facts | `src/targets/qwen3_6_35b_a3b/impl/config.h` |
| exact 934-tensor/six-resource binding, conditional residency, and immutable Text/MTP/Vision/MoE/DFlash views | `src/targets/qwen3_6_35b_a3b/impl/load/` |
| fused attention projection, fused staged GDN projection/control, sparse-MoE post-mixer leaves, leaf workspace, and graph frontier ranges | `src/targets/qwen3_6_35b_a3b/impl/variant.h`, `impl/variant.cpp` |
| fixed Text/MTP/Vision execution, planning, Program lifecycle, workspace composition, prefix/state transactions, and graph mechanics | `src/targets/qwen3_6/impl/runtime/` |
| tokenizer, template, multimodal processing, and output decoding | `src/targets/qwen3_6/impl/frontend/` |
| mathematical and explicit local-state Op contracts/implementations | `include/ninfer/ops/`, `src/ops/` |
| exact artifact and converter | [`qwen3.6-35b-a3b-artifact.md`](qwen3.6-35b-a3b-artifact.md), `tools/convert/qwen3_6_35b_a3b/` |
| artifact-native diagnostic reference | `tools/reference/qwen3_6_35b_a3b/` |

The registered 35B binder validates the complete DFlash inventory and owns a target-private
conditional materialization/model-view path. Public Engine construction currently leaves that
weight group nonresident. The DFlash sections do not claim that scheduling, state, workspace, or
execution exists.

The Python reference is diagnostic evidence, not a generated-token golden. Each production Op path
is checked against its independent mathematical oracle; equality between different numerical or
execution paths is not a runtime acceptance contract.

As descriptive provenance for the checkpoint inspected by this reference, the local `config.json`
SHA-256 is
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`, and the local
`model.safetensors.index.json` SHA-256 is
`41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`. The index and all shard
headers agree exactly: no missing, extra, duplicate, or mis-sharded tensor was found. These hashes
identify that inspection input; they are not runtime validity gates or regeneration requirements.
