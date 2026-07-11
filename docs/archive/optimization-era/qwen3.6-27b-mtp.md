# Qwen3.6-27B MTP Head 与 Prefill/Decode 交互原理

本文只解释 Qwen3.6-27B 的 MTP head 原理、结构和它与 prefill/decode 的计算关系。不讨论如何接入本仓库当前 runtime，也不把它写成实现计划。

这里的 Qwen3.6-27B 指本机 checkpoint：

```text
/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
```

这个 checkpoint 在 Hugging Face 配置中使用 `model_type = qwen3_5` / `qwen3_5_text`，源码侧的 vLLM 和 llama.cpp 也都以 Qwen3.5 family 名称组织相关代码。本文沿用“Qwen3.6-27B”称呼模型本身，沿用“Qwen3_5”称呼源码类名。

## 1. 核心结论

Qwen3.6-27B 的 MTP 不是简单多挂几个 `lm_head`，也不是让主模型一次输出多个 token。它是一个附着在主模型后的轻量 draft model：

1. 主模型正常 prefill/decode，产生目标模型 hidden state 和 immediate next-token logits。
2. 采样或接受出的下一个 token 被查 embedding。
3. MTP 把这个 token embedding 与主模型 hidden state 组合，经过一个 `fc` 和一层 full-attention decoder layer，再经 final norm 和目标 checkpoint 顶层 `lm_head.weight` 语义输出 draft logits。
4. draft token 只作为 speculative candidate。下一轮仍由主模型验证 candidate，接受前缀或拒绝后回退到主模型采样结果。

因此 MTP 改变的是吞吐路径，不应改变最终目标模型分布。这个结论依赖正确的 target-side verification：greedy 时逐 token 验证 target greedy 结果，sampling 时 rejection sampler 必须基于同一套处理后的 target 分布和实际 draft proposal 分布做接受/拒绝校正。MTP 预测错只会降低接受率。

对本 checkpoint，MTP 的关键事实是：

- `mtp_num_hidden_layers = 1`。
- `mtp_use_dedicated_embeddings = false`。
- MTP head 复用目标 checkpoint 的 token embedding 与 `lm_head.weight` 语义；因为 `tie_word_embeddings=false`，embedding 和 `lm_head` 本身并不 tied。checkpoint 里没有 `mtp.embed_tokens.weight` 或独立 `mtp.lm_head.weight`。
- MTP 自己拥有 `pre_fc_norm_embedding`、`pre_fc_norm_hidden`、`fc`、一层 full-attention decoder layer、`norm`。
- MTP layer 是 full attention，不是 Qwen3.6 主干里 48 层使用的 linear attention/GDN 层。

## 2. Checkpoint 结构事实

本地 `config.json` 中与本文相关的 text config：

| 字段 | 值 | 含义 |
|---|---:|---|
| `hidden_size` | 5120 | 主模型和 MTP hidden 维度 `D` |
| `intermediate_size` | 17408 | dense MLP 中间维度 |
| `num_hidden_layers` | 64 | 主模型 decoder 层数 |
| `mtp_num_hidden_layers` | 1 | MTP decoder layer 数 |
| `vocab_size` | 248320 | vocab 和 `lm_head` 行数 |
| `tie_word_embeddings` | false | 主模型 embedding 与 `lm_head` 不 tied |
| `mtp_use_dedicated_embeddings` | false | MTP 不使用独立 embedding |
| `num_attention_heads` | 24 | full attention query heads |
| `num_key_value_heads` | 4 | GQA KV heads |
| `head_dim` | 256 | 每个 head 维度 |
| `attn_output_gate` | true | Q projection 同时输出 query gate |
| `full_attention_interval` | 4 | 主模型每 4 层一层 full attention |

主模型 64 层中 full-attention 层号为：

```text
3, 7, 11, 15, 19, 23, 27, 31,
35, 39, 43, 47, 51, 55, 59, 63
```

其余 48 层是 linear attention/GDN 类层。MTP layer 不遵循这个 interval，而是直接是一层 full-attention layer。

本地 safetensors 中 MTP 权重如下：

| tensor | shape | 说明 |
|---|---:|---|
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | next-token embedding 侧 RMSNorm |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | hidden state 侧 RMSNorm |
| `mtp.fc.weight` | `[5120, 10240]` | concat 后从 `2D` 投影回 `D` |
| `mtp.layers.0.input_layernorm.weight` | `[5120]` | MTP decoder layer attention 前 norm |
| `mtp.layers.0.self_attn.q_proj.weight` | `[12288, 5120]` | query + output gate projection |
| `mtp.layers.0.self_attn.k_proj.weight` | `[1024, 5120]` | 4 KV heads x 256 |
| `mtp.layers.0.self_attn.v_proj.weight` | `[1024, 5120]` | 4 KV heads x 256 |
| `mtp.layers.0.self_attn.o_proj.weight` | `[5120, 6144]` | 24 query heads x 256 回投影 |
| `mtp.layers.0.self_attn.q_norm.weight` | `[256]` | per-head Q RMSNorm |
| `mtp.layers.0.self_attn.k_norm.weight` | `[256]` | per-head K RMSNorm |
| `mtp.layers.0.post_attention_layernorm.weight` | `[5120]` | MLP 前 norm |
| `mtp.layers.0.mlp.gate_proj.weight` | `[17408, 5120]` | SwiGLU gate |
| `mtp.layers.0.mlp.up_proj.weight` | `[17408, 5120]` | SwiGLU up |
| `mtp.layers.0.mlp.down_proj.weight` | `[5120, 17408]` | MLP down |
| `mtp.norm.weight` | `[5120]` | MTP output final norm |

这些 shape 与主模型 full-attention layer 的 shape 匹配。尤其是：

- `q_proj` 输出 `12288 = 24 * (256 query + 256 gate)`。
- `o_proj` 输入 `6144 = 24 * 256`。
- `k_proj` 和 `v_proj` 输出 `1024 = 4 * 256`。

## 3. MTP Head 的计算结构

令：

- `D = 5120`。
- `E[token]` 是共享 token embedding。
- `H_t` 是主模型在位置 `t` 的最终 hidden state，也就是主模型 final norm 之后、送入 `lm_head` 前的向量。
- `y_{t+1}` 是对 `H_t` 来说已经确定的 next token：在 prefill/chunked prefill 中可以是已知 prompt/query token，在 decode 中可以是新采样 token、已接受 draft、拒绝处 replacement token 或全部接受后的 bonus token。
- `M_t` 是 MTP 在该 speculative step 产生的 hidden state。

单步 MTP 的概念计算为：

```text
e = RMSNorm_mtp_embedding(E[y_{t+1}])        # [D]
h = RMSNorm_mtp_hidden(H_t or previous MTP) # [D]
u = concat(e, h)                            # [2D]
z = mtp.fc(u)                               # [D]
m = one_full_attention_decoder_layer(z)     # [D]
M_t = mtp.norm(m)                           # [D]
draft_logits = lm_head(M_t)                 # [vocab_size]
draft_token = sample(draft_logits)
```

第一步 MTP 使用主模型 hidden state `H_t`。如果要连续草拟多个 speculative tokens，后续步不再有新的主模型 hidden state；vLLM 会把上一步 MTP 输出 hidden state 作为下一步 `pre_fc_norm_hidden` 的输入，同时把刚采出的 draft token 作为下一步 embedding 输入。

因此，对 Qwen3.6-27B 这种 `mtp_num_hidden_layers = 1` 的 checkpoint，MTP 的“多 token”来自同一个 MTP block 的 autoregressive 复用，而不是 checkpoint 里有多层不同的 MTP head。vLLM 默认会把 `num_speculative_tokens` 设为 `n_predict = mtp_num_hidden_layers = 1`；如果用户把 speculative token 数设得更大，vLLM 会重复调用同一层，并提示这种方式可能降低接受率。

## 4. MTP Decoder Layer 内部

MTP 的 `mtp.layers.0` 是一层 Qwen3.5 full-attention decoder layer。它的内部顺序与主模型 full-attention 层一致：

1. attention 前 RMSNorm。
2. Q/K/V projection。
3. Q projection 同时产出 query 和 query gate。
4. Q/K 做 per-head RMSNorm。
5. 对 Q/K 应用 RoPE/MRoPE。
6. GQA attention：24 个 query heads 共享 4 个 KV heads。
7. attention output 乘以 `sigmoid(query_gate)`。
8. `o_proj` 回到 hidden size。
9. residual 连接。
10. MLP 前 RMSNorm。
11. dense SwiGLU MLP：`down_proj(silu(gate_proj(x)) * up_proj(x))`。
12. residual 连接。

MTP layer 没有主模型 linear-attention/GDN 层的 recurrent state 或 conv state。它只需要一套自己的 full-attention KV cache。

## 5. 为什么 MTP 输入是“hidden state + next-token embedding”

主模型在位置 `t` 的 hidden state `H_t` 已经表达了 prefix `x_0 ... x_t`，主模型 `lm_head(H_t)` 用来预测 `x_{t+1}`。如果要预测 `x_{t+2}`，只看 `H_t` 不够，因为还缺少已经采样出的 `x_{t+1}`。

MTP 的输入正是把两部分补齐：

```text
prefix representation: H_t
new token identity:    E[y_{t+1}]
```

然后让一层 full-attention decoder layer 学会近似“如果主模型真的接着处理了 `y_{t+1}`，下一个 hidden state 会如何变化”。MTP 不是精确重放主模型 64 层，它只是一个训练出的 draft approximation。

vLLM 的 prefill-side MTP input 构造可以用一个序列例子说明。假设 target 本轮 query token 是：

```text
target input ids:   x_a, x_{a+1}, x_{a+2}, ..., x_b
target hidden:      H_a, H_{a+1}, H_{a+2}, ..., H_b
```

MTP prefill 会把 input ids 左移一位，并在最后补入刚采样出的 token：

```text
mtp input ids:      x_{a+1}, x_{a+2}, ..., x_b, y_{b+1}
mtp hidden input:   H_a,     H_{a+1}, ..., H_{b-1}, H_b
```

于是每个 slot 都是：

```text
(target hidden at current slot, token after current slot)
```

这个 pair 的输出用于预测再下一个 token。最后一个 slot 最关键：

```text
(H_b, E[y_{b+1}]) -> draft logits for y_{b+2}
```

需要注意一个细节：vLLM 在 first MTP prefill pass 中复制 target positions，而不是把 positions 也左移或整体加一。也就是说，first MTP state 仍放在 target hidden 所在的位置索引上；MTP 产生第一个 draft token 后，后续 autoregressive draft step 才会把 position 和 MTP seq len 加一。这里的 copied positions 描述的是 MTP forward/cache slot；在 probabilistic draft sampling 中，vLLM 采样时会用 `positions + 1` 对齐 next-token 位置的 Gumbel noise。

## 6. 与 Prefill 的关系

MTP 不替代主模型 prefill。完整关系是：

1. 主模型照常处理 prompt 或本轮 chunked prefill token。
2. 主模型更新自己的 full-attention KV cache 和 linear/GDN state。
3. 主模型在需要采样的位置输出 logits。
4. sampler 从主模型 logits 得到 immediate next token。
5. MTP 使用主模型本轮 hidden states、左移后的 token ids、以及最后的 immediate next token，生成下一步 draft token。

也就是说，MTP 是在主模型已经完成本轮 forward 之后运行的 proposal path。它读主模型 hidden state，但不反向修改主模型 hidden state、KV cache 或 GDN state。

chunked prefill 有一个特殊分支：如果当前 prefill chunk 还没有产生 sampled token，vLLM 会用 `next_prefill_tokens` 作为 MTP 最后一个 slot 的 next token。这样可以维持 MTP 输入对齐。真正对 decode 有收益的是进入 generation 后、已经有 sampled/accepted token 的场景。

## 7. 与 Decode 的关系

带 MTP 的 decode 可以理解为循环执行两个阶段：

```text
target verify/accept -> mtp propose -> target verify/accept -> mtp propose -> ...
```

### 7.1 target verify/accept

假设上一轮已经有 stored draft tokens：

```text
last sampled token: y_t
draft tokens:       d_{t+1}, d_{t+2}, ..., d_{t+k}
```

下一次 target decode 不只喂 `y_t`，而是把 `y_t` 和 draft tokens 拼进同一个 target batch：

```text
target input ids: y_t, d_{t+1}, d_{t+2}, ..., d_{t+k}
```

主模型一次 forward 后，会在这些位置产生一串 target logits：

```text
logits after y_t       -> target distribution for token t+1
logits after d_{t+1}   -> target distribution for token t+2
...
logits after d_{t+k}   -> bonus distribution after all drafts
```

然后 rejection sampler 逐个验证 draft prefix：

- 如果 target 分布接受 `d_{t+1}`，继续看 `d_{t+2}`。
- 第一次不接受时，停止接受后续 draft，并用 target 分布采样出的 token 替代该位置。
- 如果所有 draft 都接受，可以再从最后一个 target logits 采样一个 bonus token。

target side 返回两个关键量：

- `num_sampled`：这轮真正进入输出序列的 token 数，包括接受的 draft、拒绝处 target sample、或者 bonus token。
- `num_rejected`：被拒绝、不能进入逻辑上下文的 draft token 数。

主模型物理上可能已经为被拒绝的 suffix 写过 cache slot，但逻辑长度、`num_computed_tokens` 和后续 slot mapping 会按 `num_rejected` 收缩。被拒绝 token 不进入后续有效上下文。

### 7.2 mtp propose

target verify/accept 结束后，MTP 根据已经确认的上下文继续提出下一批 draft。

vLLM 的 MTP proposer 会接收：

- target 本轮 hidden states；
- target input ids 和 positions；
- `last_sampled_tokens`；
- `num_sampled`；
- `num_rejected`；
- attention metadata 和 slot mapping；
- sampling 参数。

它首先重建 MTP prefill 输入：

1. 对每个 request 取 target 本轮 query。
2. 用 `query_len -= num_rejected` 得到逻辑上仍有效的 query prefix。
3. 将有效 prefix 内的 target input ids 左移一位。
4. 在最后一个 slot 写入 `last_sampled_token`，或者 chunked prefill 时写入 `next_prefill_token`。
5. positions 复制 target positions。
6. MTP current draft step 重置为 0。

实现细节上，vLLM 为了避免 CPU/GPU 同步，会让 MTP prefill 的 `input_ids` 和 `hidden_states` 物理尺寸继续保持与 target batch 相同，并通过 padding 覆盖 rejected positions。所以上面的“剔除 rejected suffix”是逻辑有效长度和 last-token index 的语义，不表示物理 tensor 被 compact。

然后 MTP forward 得到 first draft token。若只需要 1 个 speculative token，这一步结束。

如果需要多个 speculative tokens，MTP 进入自己的 autoregressive decode：

1. 把 first draft token 写成下一步 MTP input id。
2. 把 first MTP hidden state 写成下一步 MTP hidden input。
3. position 和 MTP seq len 加一。
4. 重新构造 MTP attention metadata / slot mapping。
5. 再跑同一个 MTP block，采样第二个 draft token。
6. 重复直到得到 `num_speculative_tokens` 个 draft。

这些 draft tokens 存入 request state，等待下一轮 target verify/accept。

## 8. 状态与主流程交互边界

MTP 与主模型交互的边界很窄：

```text
main model -> MTP:
  final hidden states
  accepted/sampled token ids
  positions / seq lengths / slot mapping
  draft sampling temperature and RNG seeds

MTP -> main scheduler:
  draft token ids

MTP internal -> rejection sampler:
  optional draft logits for probabilistic rejection sampling
```

MTP 不把自己的 hidden state 写回主模型，也不修改主模型 layer state。两边的状态可以这样区分：

| 状态 | owner | 作用 |
|---|---|---|
| 主模型 full-attention KV cache | target model | target prefill/decode 和 draft verification |
| 主模型 linear/GDN recurrent state | target model | target 48 个 linear attention/GDN 层 |
| 主模型 final hidden states | target model 输出，MTP 读取 | MTP first proposal 的 hidden input |
| MTP full-attention KV cache | MTP draft model | MTP 自己连续 proposal 时使用 |
| MTP autoregressive hidden buffer | MTP draft model | 多步 draft 时把上一 MTP hidden 送入下一步 |
| draft tokens | scheduler/request state | 下一轮 target verify 的候选输入 |
| draft logits | MTP speculator 内部 buffer / rejection sampler | 只在 probabilistic draft sampling 时存在，用于概率比接受率校正 |

这也是为什么 MTP layer 必须有独立 attention layer name / KV namespace。vLLM 在加载 draft model 后，会把 target attention layer names 和 draft attention layer names 分开，draft side 单独初始化 attention backend，但复用 request 的 block table / slot mapping 机制。

## 9. Greedy 与概率采样下的正确性

MTP 是 proposal model，不是 authoritative model。

在 greedy 场景中，MTP 直接给出 candidate token；target 下一轮算出自己的 greedy token 后，只有一致的前缀被接受。

在 sampling 场景中，需要维护 target distribution 与 draft distribution 的一致性。vLLM 本地配置里 `draft_sample_method` 默认是 `greedy`：MTP draft proposal 走共享接口时会复制 temperature 和 RNG seeds buffer，但 greedy 分支实际直接取 `argmax`，不使用这些 buffer；没有 `draft_logits` buffer 时，rejection sampling 把 draft probability 当成 one-hot proposal 处理。top-k/top-p 等其它 sampling 过滤也不进入 draft-side proposal。

如果 `draft_sample_method = "probabilistic"`，vLLM 才会为 MTP 分配 `draft_logits` buffer。此时 draft sampling 会保存处理后的 draft logits；target verify 阶段先通过 sampler 生成 processed target logits，再把 processed target logits、draft logits、draft sampled tokens、temperature 和 seeds 交给 rejection sampler。这样即使 draft model 分布与 target model 不同，也可以在正确实现的接受/拒绝校正下保持最终样本来自 target-side processed distribution。

所以 MTP 的质量影响：

- 接受率；
- 每轮 target forward 能吞掉多少 candidate token；
- 吞吐和延迟。

MTP 的质量不应该影响：

- target 模型定义的最终概率分布；
- 被 target 拒绝后的输出正确性。

## 10. vLLM 源码证据

vLLM 是本文最直接的源码级证据，因为它有 Qwen3.5 MTP model 和 v1 speculative decode path。

关键位置：

- `/home/neroued/vllm/vllm/model_executor/models/qwen3_5_mtp.py`
  - `60-119`：`Qwen3_5MultiTokenPredictor` 定义 embedding、`mtp.fc`、一组 full-attention `Qwen3_5DecoderLayer`、三个 RMSNorm。
  - `124-160`：MTP forward：embed input ids，分别 norm embedding 和 hidden，concat，`fc`，跑当前 MTP layer，final norm。
  - `357-445`：`Qwen3_5MTP` 包装 predictor、`lm_head`、`compute_logits`。
  - `447-460`：加载权重时把 checkpoint 里的 `mtp.` 前缀映射到 draft model 内部结构。
- `/home/neroued/vllm/vllm/model_executor/models/qwen3_5.py`
  - `124-183`：`Qwen3_5DecoderLayer` 根据 `layer_type` 选择 full attention 或 linear attention；MTP 传入的是 `layer_type="full_attention"`。
  - `480-555`、`583-704`：普通 target model 加载时跳过 `mtp.` 权重，说明 target path 和 MTP draft path 分开。
- `/home/neroued/vllm/vllm/model_executor/models/qwen3_next.py`
  - `236-323`：full-attention module 的 head、KV、Q/K norm、gate 结构。
  - `336-403`：Q projection 拆 query 与 gate，Q/K norm、RoPE、attention output 乘 gate，再 `o_proj`。
  - `406-542`：decoder layer 的 norm、attention、MLP、residual 顺序。
- `/home/neroued/vllm/vllm/config/speculative.py`
  - `469-478`：Qwen3.5/Qwen3.5-MoE target config 被改写成 MTP draft config，`n_predict = mtp_num_hidden_layers`。
  - `589-608`：`method="mtp"` 且无单独 draft model 时，使用 target model checkpoint 作为 draft source。
  - `752-819`：识别 MTP model type，默认 speculative token 数来自 `n_predict`；大于 1 时有约束和 warning。
- `/home/neroued/vllm/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`
  - `12-22`：Qwen3.5 MTP 使用 `MTPSpeculator`，继承 autoregressive speculator，`model_returns_tuple=False`。
- `/home/neroued/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
  - `136-279`：`propose` 接收 target hidden states 和 accept/reject 信息，准备 MTP prefill，再进入多步 draft。
  - `338-373`：MTP prefill 后从 last token indices 采样 first draft，并保存 MTP hidden state。
  - `374-467`：多步 MTP decode，逐步重建 attention metadata，采样并更新下一步 input。
  - `470-589`：`prepare_prefill_inputs`：剔除 rejected suffix、左移 target input ids、最后补 `last_sampled` 或 `next_prefill_tokens`、复制 positions。
  - `592-667`：`prepare_decode_inputs`：first draft token 成为下一步 input id，position/seq len 前进。
  - `669-766`：`update_draft_inputs`：保存 draft token，复制 MTP hidden state 给下一步，继续推进 position/seq len。
- `/home/neroued/vllm/vllm/v1/worker/gpu/model_runner.py`
  - `871-971`：有 draft tokens 时，target batch 会包含 sampled token 和 scheduled draft tokens，并设置 logits indices。
  - `1046-1078`：target logits 与 draft logits 交给 rejection sampler。
  - `1369-1458`：target sample 后调用 speculator `propose` 生成下一轮 draft tokens。
- `/home/neroued/vllm/vllm/v1/worker/gpu/input_batch.py`
  - `300-400`：把 last sampled token 和 scheduled draft tokens 合入 target input ids。
  - `403-449`：计算 `num_sampled` 和 `num_rejected`。
  - `452-550`：post-update 时只推进接受后的逻辑 token。
- `/home/neroued/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
  - `98-155`：target logits、draft logits、draft sampled tokens 进入 rejection sampling。

## 11. llama.cpp 源码证据与边界

`~/llama.cpp` 对本文有两类参考价值：一类是 Qwen3.5 full-attention layer 的结构证据，另一类是通用 speculative verification 流程证据。

### 11.1 Qwen3.5 full attention 结构

`/home/neroued/llama.cpp/src/models/qwen35.cpp`：

- `26-72`：主模型逐层构图，根据 layer 是否 recurrent 选择 linear/GDN 或 full attention。
- `117-195`：full attention path 中，Q projection 输出 query 与 gate；query 取前半，gate 从 offset view 取后半；Q/K norm；MRoPE；attention；`sigmoid(gate)` 乘 attention output；最后 `wo` 投影。

这与 checkpoint 中 MTP layer 的 full-attention tensor shape、vLLM 的 `Qwen3NextAttention` 逻辑一致。

### 11.2 通用 speculative verification 流程

llama.cpp 的 server/speculative path 展示了 target 验证 draft prefix 的通用模式：

- `/home/neroued/llama.cpp/common/sampling.cpp` `619-656`：`common_sampler_sample_and_accept_n` 用 target logits 采样 target token，并通过 token equality 验证 draft tokens，遇到不匹配停止；如果全部接受，再采一个 bonus token。这是 sample-and-match verification 证据，不是 draft-logits 概率比 rejection sampling 证据。
- `/home/neroued/llama.cpp/tools/server/server-context.cpp` `336-405`：生成 draft tokens 后，把 sampled token 和 draft tokens 加入下一轮 target batch。
- `/home/neroued/llama.cpp/tools/server/server-context.cpp` `2962-3025`：target batch 完成后验证/接受 draft，部分接受时恢复 checkpoint 并更新 speculative accept 统计。

这部分可以作为 speculative decode 主循环的源码参考，但它不是 Qwen3.6 MTP head 的直接实现。

### 11.3 llama.cpp 当前 Qwen3.5 MTP 支持边界

本机 `~/llama.cpp` 树里已经有泛化的 NextN/MTP tensor 名称和 GGUF 元数据槽位，但这些是全局或其它架构的基础设施，不代表 QWEN35 已经接入 MTP：

- `/home/neroued/llama.cpp/src/llama-model.h` `204-210`：`llama_layer_nextn` 中有 `eh_proj`、`embed_tokens`、`enorm`、`hnorm`、`shared_head_head`、`shared_head_norm`。
- `/home/neroued/llama.cpp/src/llama-arch.cpp` `447-452`：`blk.%d.nextn.*` tensor 名称。
- `/home/neroued/llama.cpp/src/llama-arch.cpp` `759-766`：NextN/MTP tensor 当前被标为 ignored/reserved，并作为 output tensor 保留。
- `/home/neroued/llama.cpp/gguf-py/gguf/constants.py` `840-846`、`1337-1342`：NextN/MTP tensor enum 与 name mapping。
- `/home/neroued/llama.cpp/gguf-py/gguf/gguf_writer.py` `868-869`：写出 `nextn_predict_layers` metadata。

但是，本机 `~/llama.cpp` 的 Qwen3.5 text converter 和 Qwen3.5 runtime graph 并没有把 Qwen3.6 checkpoint 的 `mtp.*` 权重接成一个可运行的 Qwen3.5 MTP proposal path。`convert_hf_to_gguf.py` 中 Qwen3.5 text model 注册在 `5435-5438`，而 Qwen3-VL vision converter 明确跳过 text/MTP 权重；`gguf-py/gguf/constants.py` 的 `MODEL_ARCH.QWEN35` tensor list 在 `1998` 起列出 target 主干 tensors，没有 NextN/MTP entries；`src/llama-model.cpp` 的 QWEN35 loader block 在 `7602` 起创建 target tensors，graph builder 在 `9046` 起选择 `llm_build_qwen35`，没有 Qwen3.5 MTP branch。Qwen3.5 graph 只构建 64 层 target 主干。

所以 llama.cpp 在本文中应被视为：

- full-attention 结构的独立证据；
- sample-and-match speculative verification 主循环的参考；
- 泛化/其它架构 NextN/MTP tensor 命名和预留结构的参考；
- 不是 Qwen3.6-27B MTP head 端到端运行实现的证据。

## 12. 端到端计算流程摘要

一次稳定的 decode loop 可以压缩成下面的伪流程：

```text
# 已有上下文 C_t，上一轮可能存着 draft tokens

1. target_input = [last_sampled_token] + previous_draft_tokens

2. target forward:
     update target KV/GDN states
     produce hidden states and logits at verification positions

3. rejection sampling:
     compare target logits with draft tokens / draft logits
     accept a prefix
     reject suffix if needed
     sample replacement token or bonus token
     update logical sequence length

4. MTP prepare:
     take valid target hidden states
     drop rejected suffix
     shift target input ids left
     append last sampled/accepted token
     copy positions

5. MTP first proposal:
     embed appended/shifted token ids
     norm embedding and target hidden
     concat -> fc -> one full-attention layer -> norm -> lm_head
     sample first draft token

6. Optional MTP autoregressive proposal:
     feed previous draft token embedding
     feed previous MTP hidden state
     advance MTP position/seq len
     repeat same MTP block

7. Store draft tokens for the next target verify pass.
```

最重要的 mental model 是：

```text
target model: authoritative verifier and state owner
MTP model:    small learned proposer
scheduler:    carries draft tokens from proposer to next verifier pass
sampler:      ensures accepted output remains target-model output
```

MTP 与主流程的交互点只有 hidden states、token ids、positions/seq lengths、draft logits 和 draft tokens。它不改变主模型层结构，不替代主模型 prefill，也不让 draft token 在未经 target 验证时进入最终上下文。
