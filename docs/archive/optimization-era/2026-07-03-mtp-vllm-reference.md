# vLLM MTP/Speculative Decoding 机制参考（细分文档 3，descriptive）

> Status: descriptive reference（事实记录，非 qus 规范；qus 规范见
> [总领文档](2026-07-03-mtp-spec-decode-overview.md) 及另两篇细分文档）。
> Date: 2026-07-03。
> 证据树: `/home/neroued/vllm`，commit `92221485aaaa4088491db3f182dd65a390fc9ac5`
> (2026-06-25, v0.23.1rc0-436)。以下行号均针对该 commit。

本文回答三个问题：vLLM 如何做 MTP 推理、draft 与主模型的关系、KV cache 与
GDN/SSM 状态在 speculative decoding 下如何管理。qus 采纳/偏离结论见总领 §10。

---

## 0. 两套 GPU model runner（读行号前必看）

该树同时存在两条 GPU 执行路径，spec decode 都已实现：

- **MRV1**：`vllm/v1/worker/gpu_model_runner.py` + `vllm/v1/spec_decode/`
  （`llm_base_proposer.py` 等）。hybrid 模型的 `num_accepted_tokens` 管道目前
  在此路径上有完整实现。
- **MRV2**：`vllm/v1/worker/gpu/model_runner.py` + `vllm/v1/worker/gpu/spec_decode/`。
  更新的重写，全 GPU 化、无 CPU 同步。`method="mtp"` 受支持
  （`vllm/config/vllm.py:2027-2032`），但 hybrid 的 `mamba_cache_mode="align"`
  不支持（`vllm/config/vllm.py:2000-2005`）；默认启用 MRV2 的架构列表不含
  Qwen3.5（`vllm/config/vllm.py:68-77`），需要 `VLLM_USE_V2_MODEL_RUNNER=1`。

两条路径的**算法约定一致**（同一 GDN metadata builder 与 triton kernels、同构的
proposer 逻辑）。本文默认引用 MRV2（结构更清晰），GDN 状态管道部分补引 MRV1。

## 1. 模型与配置

### 1.1 Qwen3.5/3.6 MTP 模块

`vllm/model_executor/models/qwen3_5_mtp.py`：

- `Qwen3_5MultiTokenPredictor.forward`（124-160）：
  `h = fc(cat(norm(embed(ids)), norm(prev_hidden)))` → 1 层
  `Qwen3_5DecoderLayer(layer_type="full_attention")` → final RMSNorm。
- 共享 embedding / lm_head：`tie_word_embeddings=false` 时独立建 `ParallelLMHead`
  并从 target checkpoint 载入（385-396, 447-460 的 `mtp.` → `model.` 重映射）。
- `current_step_idx = spec_step_idx % num_mtp_layers`（147）存在，但 MRV2 不传
  `spec_step_idx`，恒用 layer 0——对 `mtp_num_hidden_layers=1` 无影响；k>1 即
  重复使用同一层。
- `mamba_cache_mode="all"` 被拒绝（370-375）。

### 1.2 speculative 配置

`vllm/config/speculative.py`：

- `qwen3_5/qwen3_5_moe` → `model_type="qwen3_5_mtp"`，
  `n_predict = mtp_num_hidden_layers`（469-478）。
- 所有 `*_mtp` 归并为 `method="mtp"`；`use_eagle()` 包含 `"mtp"`（1109-1110）。
- `num_speculative_tokens` 默认 = `n_predict`；大于 `n_predict` 要求整除并告警
  「重复 forward 同一 MTP 层可能降低接受率」（804-819, 752-765）。
- `draft_sample_method: "greedy" | "probabilistic"`（默认 greedy，265-271）。
- MRV2 分发：`method=="mtp"` → `MTPSpeculator`
  （`v1/worker/gpu/spec_decode/__init__.py:29-32`），它是
  `AutoRegressiveSpeculator` 的薄子类：draft forward 返回单 tensor
  （`model_returns_tuple=False`），用 EAGLE loader 共享权重
  （`mtp/speculator.py:12-22`）。

## 2. Draft token 生命周期与 scheduler 记账

- 权威存放：worker GPU 状态 `draft_tokens [max_num_reqs, k]` int64
  （`v1/worker/gpu/states.py:71-77`），每步由 `propose()` 返回值 scatter 回填
  （`v1/worker/gpu/model_runner.py:1444-1458`）。scheduler 只留 CPU 计数用副本
  `request.spec_token_ids`（`v1/request.py:152, 250-252`）。
- scheduler 没有 prefill/decode 阶段概念：驱动 `num_computed_tokens` 追平
  `num_tokens_with_spec = len(all_tokens) + len(spec_tokens)`
  （`v1/core/sched/scheduler.py:390-399, 463-479`）。decode 请求一步调度
  `1 + k` 个 token；调度后 `spec_token_ids` 清空、被调度子集记入
  `scheduled_spec_decode_tokens`（582-598）。
- KV 分配含 lookahead：`num_lookahead_tokens = num_spec_tokens`（`use_eagle()`
  方法族，scheduler.py:240-244），`allocate_slots` 按
  `num_tokens + lookahead` 留槽（`v1/core/kv_cache_manager.py:389-392`）——这是
  drafter 自己 KV 写入的空间。
- 抢占：块全部释放、`num_computed_tokens=0`、spec tokens 丢弃
  （scheduler.py:1106-1127）。

## 3. Target verify batch 构造（MRV2）

`v1/worker/gpu/model_runner.py:845-1017` + `v1/worker/gpu/input_batch.py`：

- 每请求 `num_logits = num_draft_tokens + 1`；`cu_num_logits` 前缀和；
  `expanded_local_pos` = 行内 draft step 序号（`model_runner.py:892-904`）。
- `combine_sampled_and_draft_tokens` kernel 把 `last_sampled` 与 `draft_tokens`
  直接写进 `input_ids` 窗口末端，并产出 `logits_indices` = 每请求 query 窗口的
  **最后 `num_logits` 行**（`input_batch.py:299-400`）。纯 decode 请求的窗口即
  `[t0, d1..dk]`，k+1 行全部取 logits；chunked-prefill 请求只取最后 1 行。
- positions/seq_lens：`positions = num_computed + offset`、
  `seq_len = num_computed + query_len`（`input_batch.py:241-296`）。
- slot mapping 纯位置推导：`slot = block_table[pos // bs] * bs + pos % bs`
  （`v1/worker/gpu/block_table.py:239-301`）——draft 位置的 target KV 写入
  lookahead 块。

## 4. Rejection sampling（MRV2）

入口 `v1/worker/gpu/spec_decode/rejection_sampler.py:98-156`；四个 triton kernel
在 `rejection_sampler_utils.py`：

- **processed logits**：先对全部 logits 行施加完整采样管线
  （bias/惩罚/temperature/min-p/top-k/top-p → 掩为 `-inf`；
  `sample/sampler.py`），被验证的是**处理后的 target 分布**；draft 侧只有
  temperature（见 §6）。`draft_sampled = input_ids[logits_indices]`，行 `i`
  验证的 draft 是 `draft_sampled[i+1]`（`rejection_sampler.py:108-117`）。
- **greedy 分支**（`temp==0`，`rejection_sampler_utils.py:224-262`）：
  逐位比较 `target_argmax == draft`；拒绝时直接存 target argmax（无需重采样）。
- **概率分支**（261-305）：log 域比值检验
  `accept ⇔ log p(x) > log u + log q(x)`，即 `u < min(1, p/q)`；无 draft_logits
  时 q 视作 one-hot（`log q = 0`）。`u` 由 `(per-request seed, 该行绝对位置)`
  的 Philox draw 生成。
- **重采样 kernel**（312-437）：拒绝位从残差
  `log(max(p-q,0)) = p_log + log(1-exp(q_log-p_log))` 采样；one-hot draft 时
  残差 = 屏蔽被拒 token 的 target 分布（400-408）；全接受时 bonus 直接从
  target 分布采样（374-376）。Gumbel-argmax 键 `(seed, 该行位置)`。
- 输出：`sampled [num_reqs, k+1]`（前 `a` 列 = 接受的 draft，列 `a` =
  recovered/bonus），`num_sampled = a+1`；chunked-prefill 行强制
  `num_sampled = num_rejected = 0`（`input_batch.py:403-429`）。
- **随机数不变量**：逻辑槽位 `m` 的 token 无论由 target 直接采样、被 verify、
  还是被 draft 提议，噪声键都是 `(seed, m-1)`——draft 采样用 `positions + 1`
  对齐（`spec_decode/speculator.py:226-238`）。因此输出分布与接受模式无关。

## 5. num_sampled / num_rejected 回传与 KV 语义

- `post_update` kernel：`num_computed_tokens += query_len - num_rejected`；
  `all_token_ids` 追加 `num_sampled` 个 token；`last_sampled_tokens` 更新为
  列 `num_sampled-1`（`input_batch.py:452-551`；调用点
  `model_runner.py:1080-1107`）。scheduler 侧等价回退
  `request.num_computed_tokens -= num_rejected`（scheduler.py:1550-1567）。
- **被拒 KV 不清理**：位置回退后，下一步同位置重写覆盖；attention 读窗口由
  seq_lens 限界，stale 行不可见。prefix cache 永不缓存未验证块
  （`kv_cache_manager.py:324-326`）。
- hybrid 模型额外把 `num_sampled` 写入持久 `num_accepted_tokens` 张量供下一步
  GDN/mamba 元数据使用：MRV2 在 `model_states/mamba_hybrid.py:145-161`
  （`postprocess_state`，clamp ≥1），MRV1 在 `gpu_model_runner.py:1508-1515`
  （`(output_token_ids != -1).sum(dim=1)`）。

## 6. MTP proposer（`propose()` 全流程，MRV2）

`v1/worker/gpu/spec_decode/autoregressive/speculator.py`：

- **padding 不变量**（172-177 注释）：draft 的"prefill"pass 保持与 target batch
  完全同形状；被拒行成为死行——CPU 无需知道 `num_rejected`，且可复用 target 的
  attention metadata 与 slot mapping（230-242）。
- **`prepare_prefill_inputs`**（470-589，triton）：每请求
  `query_len -= num_rejected`；`draft_ids[qs+j] = target_ids[qs+j+1]`（左移）；
  末位 `draft_ids[qs+query_len-1] = last_sampled`（有采样时）或
  `next_prefill_tokens`（chunked prefill 时）；**positions 照抄 target**（行
  (token `p+1`, hidden `p`) 挂在 RoPE 位置 `p`）；`last_token_indices`、
  `current_draft_step=0`，其余按 CUDA-graph 需求补 padding。
- **draft prefill**（338-372）：跑整窗口 → 取 `last_token_indices` 行 hidden →
  采样 draft 0 → 保存该行 hidden/positions 供 decode steps。
- **draft 采样**（`speculator.py:214-240`）：greedy = 纯 argmax；probabilistic =
  Gumbel（`positions+1` 键控）且把 temperature-processed logits 存进
  `draft_logits [max_reqs, k, V]` fp32（125-133 分配）。draft 只看 temperature，
  忽略 top-k/p（252-257 注释：不影响 rejection 后的输出分布）。
- **decode steps 1..k-1**：`prepare_decode_inputs`（592-666）切到每请求 1 token
  布局：`position += 1`、`seq_len = target_seq_len - num_rejected + 1`；
  `_multi_step_decode`（374-421）每步重算 slot mapping、重建 metadata、
  forward、采样、`update_draft_inputs`（669-766：写 `draft_tokens[req,step]`、
  token/hidden 回填、position/seq_len 自增）。
- **chunked prefill**：runner 每步记录 `next_prefill_tokens[req] =
  all_token_ids[num_computed + query_len]`（`input_batch.py:205-215`）；drafter
  对每个 chunk 都跑 shifted pass（**这是 MTP 层 prompt KV 的填充方式**）；
  非终 chunk 的提案被 scheduler 丢弃（scheduler.py:1909-1913）。

## 7. Draft KV cache 与 hybrid 分组

- MTP 层是有独立 layer name 的真实 attention 层（prefix `mtp.layers.0`）；
  draft 层名集合 = 全部 attention 层 − target 层（`spec_decode/speculator.py:143-159`）。
- KV cache 分组对 hybrid 走 uniform-page-size 分组
  （`v1/core/kv_cache_utils.py:1108-1227`）：FullAttentionSpec 桶 = 16 target
  full-attn + 1 MTP = 17 层一组（1.5× 组容差启发式明确为 spec-decode drafter
  而设，1195-1203）；MambaSpec 桶 = 48 GDN 层。MTP 层**共享 full-attn 组的
  block table**，但有自己的 per-layer cache tensor。
- draft prefill 复用 target slot mapping（§6）；draft decode steps 用自己推进的
  positions 每步重算 slot mapping，落进 §2 的 lookahead 槽。
- 被拒/陈旧 draft KV 同样不清理：下一轮 pass 按位置覆盖；轮内 junk 行不可达
  （因果 mask + `seq_len = target_seq_len - num_rejected + 1` 限界）。

## 8. GDN/SSM 状态在 spec decode 下的管理（核心机制）

一句话：**不回滚、不重算——deferred commit / late selection**。verify 步把每个
位置处理后的状态快照写进 per-position 槽位；下一步 kernel 用
`num_accepted_tokens` 选择正确的初始槽。

### 8.1 状态即该层的 "KV cache"

- `MambaSpec` 携带 `num_speculative_blocks = num_speculative_tokens`
  （`model_executor/layers/mamba/abstract.py:44-58`；
  `v1/kv_cache_interface.py:629-657`：`"none"` 模式每请求预算
  `page_size × (1 + k)`）。
- 状态形状（`mamba_utils.py:213-234`）：conv
  `(conv_dim, conv_kernel-1 + num_spec)`——**conv 窗口按 k 加宽**；ssm
  `(HV, V, K)` 不加宽，**用多块槽位**。dtype：conv = cache dtype（默认模型
  dtype），ssm 可由 HF `mamba_ssm_dtype` 钉住（`models/config.py:603-627`）。
- 无 prefix caching 时 `mamba_block_size = max_model_len`，每请求 mamba 块表行
  恰为 `[主槽, spec 槽 ×k]`（`models/config.py:404-461`；
  `gpu_model_runner.py:6987-6996`）。块 0 是保留 null 块，kernel 对
  `state_idx <= 0` 直接跳过（CUDA-graph padding 安全）。

### 8.2 metadata builder

`v1/attention/backends/gdn_attn.py`：

- `GDNAttentionMetadata`（41-79）：`num_spec_decodes`、
  `spec_state_indices_tensor [batch, k+1]`（= 块表前 k+1 列）、
  `spec_sequence_masks`、`spec/non_spec_token_indx`（混合 batch 的 token 重排
  gather 索引）、`num_accepted_tokens [batch]`。
- spec 行判定：`num_decode_draft_tokens_cpu >= 0`（-1 = 非 spec 行；chunked
  prefill 中的请求即使有 draft 也标 -1）（189-209；MRV1 侧
  `gpu_model_runner.py:2176-2178`）。
- 普通 decode 与 spec decode 不共存：有 spec 行时普通 decode 重分类为 prefill
  （243-251，断言 407-411——prefill kernel 对带初始状态的 1-token 序列结果
  相同）。
- `num_accepted_tokens` 传递链（MRV1）：rejection 输出
  `(output_token_ids != -1).sum(dim=1)` → 持久 GPU 张量 →下一步
  `_prepare_inputs` 对齐/补 1 → 仅传给 mamba/GDN builder
  （`gpu_model_runner.py:1508-1515, 2036-2070, 2398-2416`）。

### 8.3 kernel 契约

- **SSM（fused recurrent）**：`fla/ops/fused_sigmoid_gating.py`（与
  `fused_recurrent.py` 同构）。`IS_SPEC_DECODING` 时：
  初始状态从槽 `ssm_state_indices[n, num_accepted-1]` 读（102-120）——即上一步
  最后一个**被接受** token 之后的快照；随后对本步 `T = 1+k` 个 token 逐个更新，
  处理完第 `i_t` 个 token 就把状态写进槽 `ssm_state_indices[n, i_t]`
  （156-170，`ht` 与 `h0` 同池，`inplace_final_state=True`）。**无回滚、无
  拷贝、无重算**；被拒槽位留死数据等着被覆盖。
- **conv（causal_conv1d_update）**：`mamba/ops/causal_conv1d.py:836-933`。
  单槽 + 加宽窗口：读 taps 于 offset `num_accepted-1`；整窗重写为
  `[old[offset+1 : offset+w-1], x_0..x_T-1]`（滑一格 + 追加全部新输入，被拒
  输入留在缓冲里，靠下一步的读 offset 逻辑剔除）。有效
  `state_len = w-1+(T-1)`（1181-1184）。
- **prefill（chunked）**：`chunk_gated_delta_rule`（FLA chunk=64）+
  `causal_conv1d_fn`；初始状态按 `prefill_state_indices` gather、
  `~has_initial_state` 清零、终态 scatter 回槽
  （`qwen_gdn_linear_attn.py:1504-1532`）。spec 行的 token 用稳定 argsort 从混合
  batch 中 gather 出来单独走 spec kernel，输出再 scatter 回原 token 序
  （`_forward_core`，1329-1576）。
- **驱动关系**：spec 路径始终走串行逐 token 的 recurrent kernel（不是 chunked
  kernel），与逐 token decode 的数学顺序一致。

### 8.4 target full-attn KV / MTP KV

无特殊机制：位置回退 + 覆盖（§5、§7），与非 hybrid 模型相同。MTP 层没有任何
GDN 状态（纯 full attention）。

## 9. CUDA graph / uniform batch

- decode 全图捕获要求 uniform query length：`decode_query_len = k+1`；capture
  尺寸取 `k+1` 的倍数（`config/compilation.py:1462-1502`）。GDN 只支持
  UNIFORM_BATCH 级全图（`gdn_attn.py:83`；builder 内部把 spec 元数据拷进持久
  预分配缓冲并按 `NULL_BLOCK_ID`/False/1 padding，418-462）。
- speculator 有两套图：k+1 token 的 draft-prefill 图（复用 target 注意力捕获
  态）与 query_len=1 的 draft-decode 图（`autoregressive/speculator.py:73-94`）。
- §6 的 padding 不变量正是让 draft batch 形状与 target 恒等、全程无 CPU 同步的
  关键。

## 10. 其它工程事实（qus 可忽略但需知晓）

- 采样参数限制：draft 侧只认 temperature；无 per-request spec 开关；structured
  output 经 grammar 校验 draft（无效补 -1，必然被拒）。
- `rejection_sample_method="synthetic"`：按配置接受率随机接受，用于基准测试。
- MRV1 的 `disable_padded_drafter_batch` 仅 MRV1 有效；MRV2 恒 padded。
- `mamba_cache_mode="align"`（prefix caching）给状态做块边界快照拷贝
  （`v1/worker/mamba_utils.py:26-192, 678-741`），qus 无 prefix caching，
  不采纳。
