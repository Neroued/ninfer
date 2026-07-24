# Qwen3.6-35B-A3B DFlash 引擎集成设计

本文是 DFlash 引擎集成的第一版审阅稿。它承接已经完成或正在完成的 DFlash Op
资格验证，定义从可调用算子到 `.ninfer` Engine 产品路径之间的启动配置、权重驻留、
target feature capture、proposal schedule、状态事务、prefix reuse、CUDA Graph 和验收边界。

本文不重新定义 DFlash 数学语义和单 Op 合同。相关权威仍为：

- [`qwen3.6-35b-a3b-model.md`](qwen3.6-35b-a3b-model.md)；
- [`qwen3.6-35b-a3b-artifact.md`](qwen3.6-35b-a3b-artifact.md)；以及
- [`qwen3.6-35b-a3b-dflash-op-checklist.md`](qwen3.6-35b-a3b-dflash-op-checklist.md)。

本设计的目标是先形成一个正确、完整、可测量的显式组合路径。本阶段不实现 Op
checklist Section 6 中记录的后续融合。

## 1. 已冻结的产品决策

### 1.1 Speculative backend 互斥

一个 Engine 实例在启动时只选择以下三种模式之一：

| Backend | Draft 生成方式 | Draft 范围 | Target verify/accept |
|---|---|---:|---|
| `none` | 无 | 0 | ordinary target decode |
| `mtp` | 单层 MTP autoregressive proposal | `1..5` | 共享 speculative transaction |
| `dflash` | 六层 masked-block parallel proposal | `1..15` | 共享 speculative transaction |

`mtp` 与 `dflash` 不同时加载、不同时分配状态、不同时准备 CUDA Graph，也不在请求中动态
切换。两者共享 speculative round 的 target verification、accept/correction、target state
publication、输出统计和 host pending-round 协议，但拥有不同的 proposal model、proposal
state 和 proposal schedule。

### 1.2 DFlash 与 Vision 互斥

DFlash 是 Text-only backend。选择 `dflash` 时：

- `enable_vision=true` 是启动配置错误；
- Vision 权重不上传到设备；
- MTP decoder weights 和 MTP KV/state 不上传或分配；
- optimized proposal head 仅在 `proposal_head=optimized` 时上传，与是否加载 MTP
  decoder 无关；
- public Frontend 不接受包含 image/video 的请求。

这不是“一个请求临时退回非 DFlash Vision”的动态策略。需要 Vision 的进程应创建一个
未启用 DFlash 的 Engine。

MTP 继续支持当前 Text/Vision 路径。`none` 也继续允许独立选择 Vision。

### 1.3 Proposal head

MTP 和 DFlash 都支持两种 proposal-head 配置：

- `full`：使用共享的完整 Q6 target output head `[248320,2048]`，只对 proposal
  columns 计算 logits，并在 `valid_rows=248077` 上 argmax；
- `optimized`：使用 Q4 shortlist head `[131072,2048]`，先在 131072 个 shortlist
  rows 上 argmax，再通过 `text/draft_head_token_ids [131072]` 映射回完整 token id。

optimized head 是 backend-neutral 的 proposal 资源，不是 MTP decoder 的组成部分。它
给出的 shortlist greedy token 是合法 draft；其结果不要求与 full-head argmax 相同，
最终输出正确性仍由共享 target verify/accept 保证。选择 DFlash+optimized 时加载
shortlist weight 和 token-id map，但不加载 MTP layer weights 或 MTP KV/state。

### 1.4 DFlash cache dtype

六层 DFlash context K/V 固定为 BF16。`EngineOptions::kv_cache` 仍只选择 target Text KV
的 BF16 或 INT8-G64 存储，不改变 DFlash context 格式。

### 1.5 Prefix boundary 保存

前五层 local-attention cyclic K/V 在 assistant boundary 处保存一份完整 BF16 快照：

```text
5 layers × K/V × 8 heads × 128 × 4096 × BF16 = 80 MiB
```

恢复 boundary 时逐 bit 复制回 active local cache。第六层 full-context cache 不做快照，
只恢复 logical frontier。

80 MiB snapshot只服务跨请求的assistant-boundary restore。Steady speculative round在host
最终确认 licensed token prefix 前不写DFlash context；因此partial-terminal truncation不需要
per-round cyclic-cache undo。

## 2. 当前实现基础与缺口

当前代码已经具备以下可复用基础：

- `SpeculativeRoundState` 的公共几何支持 `K=15`；
- `speculative_prepare_verify_inputs`、target verify、
  `speculative_accept_greedy_drafts` 和 accepted-hidden selection 已经与 MTP proposal
  schedule 分离；
- target causal verification 和 GDN snapshot publication 已覆盖 `A=0..15`；
- Program 已有 `PendingKind::Speculative`、host licensed-token publication、
  `resolve_pending` 和 speculative statistics；
- Text layer runner 已有编译期 Tap 接口，并在每层 post-MLP residual 处提供
  `AfterMlp` 观察点；
- 35B artifact binder 已验证并可条件物化 DFlash 权重。

仍缺少的引擎能力是：

1. public option 仍把非零 `draft_tokens` 等同于 MTP；
2. 35B `plan_load` 仍固定 `dflash=false`；
3. Program 只接收 ordinary/MTP family model view，没有可执行的 DFlash view；
4. persistent layout 只有 target/MTP state，没有六层 DFlash cache和boundary snapshot；
5. workspace planner 不包含 target-feature capture 和六层 proposal route；
6. production Text schedule 使用 `NullTap`，diagnostic tap 不能作为 Graph-safe feature
   capture；
7. speculative controller、graph vectors 和 frontier 字段仍带有 MTP 专属所有权；
8. prefix restore 尚不知道 DFlash local cyclic cache。

这些都是集成工作，不要求增加新的 proposal 数学 Op。

## 3. Public option 与启动配置

### 3.1 Engine API

建议将 backend 作为 `SpeculativeOptions` 的显式字段，而不是从 draft 数量推断：

```cpp
enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};
```

这是 project-owned API，不保留“非零 draft 数量隐式选择 MTP”的第二条路径。

### 3.2 配置验证

验证在 materialization plan 建立前完成：

| Backend | 合法配置 | 非法配置 |
|---|---|---|
| `none` | `draft_tokens=0`, `proposal_head=full` | 非零 draft、optimized head |
| `mtp` | `draft_tokens=1..5`, full/optimized head | DFlash 范围、缺失 MTP 权重 |
| `dflash` | 35B、`draft_tokens=1..15`、full/optimized head、Vision off | 27B、Vision on |

所有非法组合在 Engine construction 早期报错，不静默降级，不额外上传不可能被调用的权重。

### 3.3 CLI 与 server 启动参数

建议统一为：

```text
--spec mtp|dflash
--draft-tokens N
--lm-head-draft
```

不传 `--spec` 即关闭 speculative decoding；选择 backend 后必须显式传
`--draft-tokens`。不传 `--lm-head-draft` 使用 full head，传入后使用 optimized head，
保持当前 optimized LM-head 命令不变。例如：

```text
--spec mtp --draft-tokens 3 --lm-head-draft
--spec dflash --draft-tokens 15 --lm-head-draft
```

CLI 和 server 使用同一个转换函数生成 `EngineOptions::speculative`，不分别维护范围和
互斥规则。`--draft-tokens` 或 `--lm-head-draft` 在缺少 `--spec` 时是配置错误。
OpenAI/Anthropic request schema 不增加 backend 字段；backend 仍是进程启动时冻结的
Engine 配置。

最终实现时同步更新 executable `--help`、`docs/cli.md`、`docs/serving.md`、README 和启动
配置测试。

## 4. 条件加载与 immutable model view

### 4.1 35B 加载矩阵

Artifact 的完整 inventory 始终被 binder 验证；只有选中功能的 payload 放到 Device：

| Backend | Base Text | MTP | optimized proposal head | DFlash | Vision |
|---|:---:|:---:|:---:|:---:|:---:|
| none, Vision off | device | metadata | metadata | metadata | metadata |
| none, Vision on | device | metadata | metadata | metadata | device |
| MTP full | device | device | metadata | metadata | optional |
| MTP optimized | device | device | device | metadata | optional |
| DFlash full | device | metadata | metadata | device | metadata |
| DFlash optimized | device | metadata | device | device | metadata |

DFlash 的 token embedding 继续别名到 Base Text 已上传的矩阵。full proposal head
别名到 `text/output_head`；optimized proposal head 复用 artifact 中唯一的
`text/draft_head` 和 `text/draft_head_token_ids`，均不产生重复 payload。

27B binder 不注册 DFlash 执行能力。选择 DFlash 时在 target package admission 阶段直接
拒绝，不引入缺失权重 fallback。

### 4.2 Model-view 所有权

Program 必须接收一个完整 immutable exact-target model view，其中最多存在一个 proposal
backend：

```text
Exact 35B model view
├── common Text weights
├── optional optimized proposal weights
├── optional MTP weights
├── optional DFlash weights
└── optional Vision weights
```

Family runtime 持有 semantic views 和调度算法；35B package 继续负责精确 tensor
binding、量化格式、DFlash 六层 payload 和已填充 view。不要把 DFlash weights 作为与
Program model 生命周期无关的第二个裸引用，也不要增加 virtual backend、plugin registry
或字符串驱动 dispatch。

构造时验证：

- frozen startup features 与 optional views 一致；
- `mtp` 与 `dflash` 不同时存在；
- optimized proposal view 可以与 MTP 或 DFlash backend 组合，但不能在 `none` 模式存在；
- DFlash view 只在 35B variant 可用；
- DFlash view 存在时 Vision view 不存在；
- inactive weights 没有 Device placement。

## 5. Runtime 所有权和调度边界

### 5.1 Family runtime 继续拥有

- `SequencePlan<Variant>` 和 `RequestPlan<Variant>`；
- speculative target input preparation；
- target verify、accept/correction 和 target state publication；
- generated-token pending/publication 协议；
- ordinary/speculative round 选择；
- prefix reuse 生命周期；
- workspace composition 和 CUDA Graph orchestration；
- MTP 与 DFlash 的 proposal schedule 调用顺序。

### 5.2 Exact 35B variant 继续拥有

- DFlash exact weight view；
- 六层、hidden/intermediate/head/window 等编译期 profile；
- feature layer ids `[1,6,11,16,22,27,32,37]`；
- target-private projection leaf payload；
- DFlash graph frontier ranges 和 Program instance bytes。

### 5.3 Ops 继续拥有

所有数学闭合操作仍在 `src/ops`。Program 不实现新的 GEMM、attention、norm 或 cache
kernel。DFlash schedule 组合已经资格验证的：

- `prepare_masked_block`；
- W8 embedding；
- W8 `linear`、`linear_pair`、`attn_input_proj`、`linear_swiglu`、`linear_add`；
- plain RMSNorm 和 D128 RoPE；
- `swa` 和 `bidirectional_gqa_attention`；
- `kv_cache_append_prefix`；
- Q6/Q4 `linear`、`argmax` 和 `proposal_remap_token_ids`。

## 6. Frozen sequence state

### 6.1 公共 speculative state

以下内容继续由 backend 共用：

- `draft_window=K`；
- `draft_tokens [K]`；
- target input ids/positions `[K+1]`；
- target argmax、verify hidden 和 logits；
- accepted drafts `A`、produced count `C=A+1`；
- round output tokens；
- statistics；
- target GDN snapshot slots；
- pending round 和 host output buffer。

`drafts_ready` 应保留为 backend-neutral 语义：

```text
drafts_ready ==
    draft_tokens 属于当前 target execution frontier E
```

必要时同时记录 `proposal_frontier`，禁止只用一个 bool 掩盖 frontier 不一致。

### 6.2 Backend-exclusive state

`none` 不分配 proposal state。

MTP 继续分配：

- one-layer MTP KV；
- alignment ids、position 和 AR hidden；
- MTP graph variants；
- MTP-specific proposal workspace。

DFlash 分配：

- 五层 BF16 cyclic K/V，容量 4096；
- 一层 BF16 full K/V，容量 `max_context`；
- 五层 local K/V 的 80 MiB boundary snapshot；
- DFlash context frontier；
- boundary snapshot frontier/validity；
- pending context update的feature、positions和count；
- proposal frontier/anchor/epoch；
- DFlash graph variants。

MTP 与 DFlash state 使用显式可选分支，不能同时构造。

### 6.3 DFlash frontier

至少区分：

```text
committed_target_frontier E
dflash_context_frontier
dflash_proposal_frontier
pending_context_base/count/valid
```

无pending context update时要求：

```text
dflash_context_frontier == E
drafts_ready =>
    dflash_proposal_frontier == E
    && dflash_proposal_anchor == pending_anchor
    && dflash_proposal_epoch == request_epoch
```

continuing `resolve_pending`发布新的committed target `E`后，可以存在一个尚待下一次GPU
提交消费的context update：

```text
pending_context_valid
dflash_context_frontier == pending_context_base
pending_context_base + pending_context_count == E
proposal_valid == false
```

下一次steady或ordinary Graph必须先消费该update并恢复
`dflash_context_frontier==E`。Pending speculative round也可以包含target已物化、尚未经过
host output policy最终确认的状态；verify feature和positions保持未发布。DFlash只提交
host最终保留的prefix，不存在provisional cyclic-cache write。

## 7. Target feature capture

### 7.1 精确 capture 点

对每个 target input column，捕获 Text layer：

```text
[1, 6, 11, 16, 22, 27, 32, 37]
```

各自完成 post-MLP residual update 后的完整 BF16 hidden `[2048,T]`。按上述顺序写入固定
切片：

```text
target_features [16384,T]
slice i = layer feature_layers[i]
```

不捕获 mixer output、next-layer input norm、final norm hidden 或 diagnostic 转换值。

### 7.2 Production tap

复用现有 `TextContext::run_layers<Tap>` 编译期结构，增加 production DFlash feature sink：

```text
NullFeatureSink       # none/MTP production
DFlashFeatureSink     # DFlash prefill/verify/ordinary
Diagnostic sink       # 现有诊断
```

Feature sink按启动时冻结的backend选择，而不是按round kind选择。none/MTP的target forward
实例化Null sink；DFlash的prefill、speculative verify和ordinary fallback全部实例化
`DFlashFeatureSink`。不要在每个非DFlash layer热路径里增加运行时callback。

Sink 在 `AfterMlp`：

1. 对八个选中层执行 pitched 2-D device copy或等价strided pack到固定slice；每个
   `[2048,T]` source写入`[16384,T]`中的row slice，不能误用连续1-D copy；
2. 同一 stream 保证 copy 在后续 layer 原地修改 hidden 前完成；
3. 其他层不写 feature buffer；
4. 一次 target call 结束时验证八个 slice 都已发布。

现有 prefill 在一个 `TextContext::prefill` 调用内部循环多个 chunk，因此只增加
`AfterMlp` copy 不足以完成集成。production sink 还需要一个明确的 chunk-complete
边界：

```text
run target layers for one chunk
    ↓
feature_sink.target_chunk_complete(positions, phase)
    ↓
prefill: 立即构造并append该chunk的DFlash context
verify:  封存capture和原始target positions，跨host decision保存
ordinary: 构造并append确定提交的单列context
```

Prefill 必须在进入下一个 chunk、feature buffer 被覆盖前消费当前 capture。Verify capture
则必须跨过 target final norm、logits、argmax、shared accept和host output policy保持有效。
下一轮开始前，或terminal resolve时，使用最终确认的count和accept前保存的
`target_positions[0:count]`构造并提交context prefix；不能从accept后已推进的`io.pos`
重新生成positions。

diagnostic tap 与 production feature sink 可以在测试构建中组合，但 diagnostic 模式继续
禁用 product CUDA Graph selection。

### 7.3 Shape 和生命周期

Feature workspace 的最大形状是：

```text
[16384,prefill_chunk]
```

默认 `prefill_chunk=1024` 时为 32 MiB。Target verify 使用同一地址的
`[16384,K+1]` view，最大只占 512 KiB。每个 target prefill chunk 或 verify call 的
feature 必须在该地址被下一次覆盖前完成 DFlash context construction。

该 buffer 不能是 `target_verify` 结束时随 `work_.reset()` 失效、随后被 sampling workspace
覆盖的普通临时 allocation。DFlash mode 的 workspace 应显式分为：

```text
round-live fixed regions
├── target feature capture
├── pending target positions/count
└── graph-stable proposal tensors

operator scratch subarena
└── 可以在每个Op或stage后reset
```

两部分来自同一个预规划 workspace allocation，但拥有不同 reset 边界。CUDA Graph capture
使用固定 region offsets，不依赖一次调用碰巧相同的动态 allocation 顺序。

## 8. DFlash context construction

对一个完整 committed target chunk，或一个尚待设备端选择前缀的 verify block：

```text
target_features [16384,T]
    │
    ├─ W8 feature projection [2048,16384]
    │
    └─ plain RMSNorm
         ↓
       c [2048,T]
```

每个 DFlash layer 使用自身 stored QKV parent 的 K/V row views：

```text
k_raw, v = linear_pair(c, W_k, W_v)
k_norm   = plain_head_rmsnorm(k_raw)
k        = rope_d128(k_norm, absolute_positions)
append selected prefix of (k,v)
```

规则如下：

- target prefill chunk 全部 committed，append count 为该 chunk 的 `T`；
- ordinary target step append count 为 1；
- speculative verify只保存`target_features`、原始`target_positions`和
  `produced_count=C=A+1`，不立即写DFlash cache；
- continuing round在下一次GPU提交开始时使用上一轮device `produced_count=C`提交；
- terminal round在`resolve_pending`中使用host最终保留数`R`执行一次slow-path exact commit；
- rejected verify-feature tail 不进入任何 DFlash context cache；
- local layers按绝对位置写 cyclic slots；
- full layer按绝对位置写 full cache；
- temporary masked-query K/V 永不进入 context cache。

当前阶段保持 `c` materialized 一次并 fan out 到六层，不实现
`context_kv_append` future fusion。

该延迟提交把host decision放在cyclic write之前。非terminal round按产品合同必须保留全部
licensed tokens，因此下一轮提交`C`；terminal round只提交最终保留的`R`。80 MiB boundary
snapshot保持不变，不承担round transaction。

## 9. Initial prefill 与第一组 proposal

Text-only `begin` 在 DFlash 模式执行：

1. target prefill 按 chunk 运行，并捕获八层 feature；
2. 每个 chunk 构造 `c`，更新六层 DFlash context；
3. 如果 chunk 精确结束在 assistant boundary，保存五层 local cache 的 80 MiB 快照；
4. target final hidden 通过现有 sampling 产生第一个未处理 anchor；
5. 在当前 context frontier `L=prompt_tokens` 上运行一次 DFlash proposal；
6. 将第一个 anchor 返回给 product controller，并令 proposal 对 frontier `L` ready。

当前 target prefill 已为 GDN boundary snapshot 把 chunk 截断到 boundary。DFlash snapshot
必须复用同一个精确 chunk 边界，在该 chunk 的 context append 完成后、继续 suffix 前执行。

令`B=K+1`，frontier `E`表示已由target处理的token数。当前verify输入位置为
`[E,E+K]`，因此：

```text
can_verify(E,K) = E + B <= target_capacity
```

确认本轮`C`个token后，新frontier为`P=E+C`，下一proposal临时位置为`[P,P+K]`：

```text
can_prepare_next(P,K) = P + B <= dflash_position_limit
```

steady Graph在进入时已经知道上一轮最终保留全部`C`，若`can_prepare_next`或随后的
`can_verify`不成立，则进入ordinary tail而不构造无机会消费的proposal。不能复制MTP的
`E+2K`判断。

## 10. DFlash proposal schedule

对 target context frontier `L`、draft count `K`、block size `B=K+1`：

1. `prepare_masked_block` 建立：

   ```text
   ids       = [anchor, MASK, ..., MASK]
   positions = [L, L+1, ..., L+K]
   ```

2. W8 embedding 生成 `u_0 [2048,B]`；
3. 依次运行六层：

   ```text
   n = plain_rmsnorm(u)
   q_raw, k_raw, v = attn_input_proj(n)
   q = rope(plain_head_rmsnorm(q_raw))
   k = rope(plain_head_rmsnorm(k_raw))

   layer 0..4: a = swa(q, k, v, persistent_context)
   layer 5:    a = bidirectional_gqa_attention(q, k, v, persistent_context)

   u += attention_output(a)
   m = plain_rmsnorm(u)
   h = linear_swiglu(m)
   u += mlp_down(h)
   ```

4. 只取 mask columns `1..B-1`；
5. final plain RMSNorm；
6. 按启动时冻结的 proposal-head route 生成 `draft_tokens [K]`：

   ```text
   full:
     logits = Q6_linear(text/output_head, hidden)       # [248320,K]
     drafts = argmax(logits, valid_rows=248077)

   optimized:
     short_logits = Q4_linear(text/draft_head, hidden) # [131072,K]
     short_rows = argmax(short_logits, valid_rows=131072)
     drafts = proposal_remap_token_ids(
         short_rows, text/draft_head_token_ids)
   ```

Anchor column不产生 proposal token。DFlash proposal 是 same-position greedy prediction，
不使用 target sampling 配置，也不生成 proposal probability。

full route 在 target accept 已经消费完本轮 target logits 后覆盖 shared
`io.logits[:,0:K]`；下一轮 target verify 会在读取完 `draft_tokens` 后重新写满
`io.logits[:,0:K+1]`。optimized route 使用 contiguous shortlist-logits workspace，
因为 shared full-vocabulary logits 的列 stride 不适合作为 `[131072,K]` 连续输出。
两条 route 都复用现有 proposal-head 调度语义。

## 11. 共享 verification transaction

MTP 和 DFlash 都调用同一个 target transaction：

```text
anchor + K drafts
    ↓ speculative_prepare_verify_inputs
target causal verify [K+1]
    ↓
target logits / argmax / hidden / target KV / GDN snapshots
    ↓ speculative_accept_greedy_drafts
A accepted drafts
C = A + 1 produced/licensed tokens
next anchor = correction or bonus
```

公共 transaction 继续负责：

- target input ids 和 absolute positions；
- target verification；
- greedy/stochastic target acceptance；
- target `io.pos` 更新；
- target GDN initial slot；
- accepted target hidden 选择；
- round tokens、count 和 statistics。

Backend-specific continuation发生在 accept 后：

### MTP continuation

使用 accepted target hidden 同步 MTP state，并 autoregressively 生成下一组 drafts。

### DFlash continuation

1. shared accept只发布licensed tokens和`produced_count=C`；
2. verify捕获的target features和原始positions被标记为pending context update；
3. host output policy调用`resolve_pending`确定最终保留数；
4. continuing round要求保留全部`C`，下一次steady Graph首先提交该pending context；
5. terminal round只在slow path提交最终保留的`R`；
6. context提交后，使用当前pending anchor运行masked-block proposal，再进入target verify。

continuing steady Graph中的append count继续使用device-resident `C`，不由host读取后选择
launch长度。只有已经结束、不会再进入steady Graph的terminal slow path使用host `R`。
Proposal identity绑定`(frontier, anchor, request_epoch)`，不能只用`drafts_ready`布尔值。

## 12. Ordinary fallback

以下情况使用 ordinary target step：

- remaining output budget 不足以许可完整 speculative return；
- context capacity 不满足当前 backend 的安全范围；
- begin因capacity没有准备第一组DFlash proposal。

DFlash-enabled Program 的 ordinary Graph先消费可能存在的上一轮pending context update，
再运行ordinary target step，捕获并提交其确定的一列feature，使DFlash context保持与target
一致。Active请求一旦因为budget或capacity进入ordinary tail，本请求不再返回speculative
route，也不再支付下一组DFlash proposal成本；可以只保留：

```text
dflash_context_frontier == E
proposal_valid = false
```

下一次`begin`若配置和capacity允许，再从新请求的pending anchor构造proposal。不要设计
同一Active请求中的隐式re-arm，也不要为了省一次proposal而让DFlash context落后于可复用
target frontier。

## 13. Prefix reuse 和两类回退

### 13.1 Exact-frontier append

`AppendAtFrontier` 要求 incoming prompt 匹配当前 target frontier，并同时验证：

```text
dflash_context_frontier == reuse_base
```

active local/full cache 原地继续；不恢复 boundary snapshot。

### 13.2 Assistant-boundary restore

保存 boundary 时：

1. target Text KV/full DFlash KV 已包含 prefix；
2. target GDN state 和 boundary hidden 使用现有 checkpoint；
3. 五层 active local K/V 完整复制到 80 MiB snapshot；
4. 记录 `boundary_frontier` 和 valid bit。

恢复时：

1. 验证 incoming prompt identity 与 boundary prefix 一致；
2. 复制 80 MiB local K/V snapshot 回 active cache；
3. 将 full DFlash cache logical frontier 降到 boundary；
4. 恢复 target GDN/hidden 和现有 target/MTP-independent state；
5. 令 DFlash proposal invalid；
6. 对新 prompt suffix 重新 target prefill、capture feature、更新 DFlash context；
7. 在新 prompt frontier 建立 proposal。

snapshot 每次请求只保留最新一个 assistant boundary，与当前简单 prefix-caching 语义一致。

### 13.3 Partial terminal exact commit

Speculative verify后，target state已经物化`C`个候选输入位置，但DFlash context尚未写入。
任何terminal resolve都执行context-flush slow path；host stop policy最终保留`R<=C`个
licensed token时：

1. 复用公共target rollback，将target KV logical frontier、GDN slot、tail hidden、
   ledger和identity提交到`base_E+R`；
2. 从保存的verify features和原始target positions构造六层K/V；
3. 仅把prefix `0..R-1`写入DFlash local/full context；
4. 发布`dflash_context_frontier=base_E+R`；
5. 丢弃pending context update和旧proposal，将Program留在Resident状态。

因为host decision发生在cyclic write之前，没有被错误覆盖的旧row，也不需要undo log。
异常或取消若使Program进入Invalid，可以直接丢弃pending capture；下一次请求执行完整
reset。

## 14. CUDA Graph 设计

### 14.1 Graph body

DFlash不照搬MTP在host decision前准备下一组draft的组织。Steady-state Graph按以下顺序
旋转：

```text
previous round's confirmed pending context
  → context construction and exact prefix append
  → DFlash proposal for current anchor
  → target verify with feature capture
  → shared accept
  → leave captured feature/positions/count pending
```

第一组proposal在prefill后eager建立，因此第一次decode使用一个只消费ready proposal的
initial verify Graph；之后进入上述steady Graph。terminal resolve使用不生成proposal的
context-flush slow path。这样每个continuing output round仍只有一次GPU提交，但不会在host
确认前写cyclic cache。

完整round benchmark必须用稳定态边界计入context update、proposal和target verify，不能因
工作从上一轮尾部旋转到下一轮头部而漏算。

### 14.2 Graph variants

一个 Program 的 backend 和 `K` 启动时固定，因此只捕获该 `K` 的 graphs。整个产品通过
分别创建 Engine 覆盖 `K=1..15`。

对host execution frontier范围`[minE,maxE]`，上一轮count满足`C∈[1,B]`，proposal
context envelope覆盖`[minE+1,maxE+B]`。Graph range必须同时落在target attention、SWA
和full-context attention各自固定dispatch route内。

DFlash graph frontier ranges 是以下 route boundary 的并集：

- target causal verification；
- DFlash layer 5 full-context attention；
- local SWA 的短上下文/满窗口 route；
- capacity-safe steady range。

W8/Q6 route只依赖启动时固定的`K/T`，不是frontier分界。Capacity tail由host scheduler
进入ordinary path，不作为同一steady Graph中的数据依赖分支。

前五层 cyclic wrap 由绝对位置和 kernel 内部映射处理，不为每个 ring offset 捕获新 graph。

### 14.3 Stable storage

以下地址在 capture/replay 间固定：

- shared speculative controls；
- target-feature slices；
- context projection和K/V candidate；
- masked ids/positions；
- six-layer temporary q/k/v/attention/MLP buffers；
- full route 的 shared logits，或 optimized route 的 shortlist logits，以及 argmax/remap
  输出；
- DFlash active caches；
- 跨host decision保存的pending feature、positions和count。

Boundary 80 MiB snapshot/restore 属于 request prefill/reuse eager 路径，不进入 steady-state
decode graph。

## 15. Memory planning

### 15.1 Persistent additions

| Region | 大小 |
|---|---:|
| 五层 active local BF16 K/V | 80 MiB |
| 五层 boundary snapshot | 80 MiB |
| 一层 full BF16 K/V | `4096 × max_context` bytes |
| DFlash frontier/control scalars | 可忽略 |

Full cache 在 `max_context=262144` 时为 1 GiB；在 2048 时为 8 MiB。

### 15.2 Workspace additions

至少包含：

- target features `[16384,prefill_chunk]`，默认最大 32 MiB；
- normalized context `[2048,prefill_chunk]`；
- per-layer context K/V candidates；
- masked block hidden `[2048,B]`；
- Q/K/V 和 attention reductions；
- dense MLP intermediate `[6144,B]`；
- optimized-head-only shortlist logits `[131072,K]`，`K=15` 时 3.75 MiB；
- 各 Op 报告的 route-private workspace。

Shared RoundState 中已有的 `io.logits [248320,K+1]` 服务 target verify；full-head DFlash
proposal 也复用它。`K=15` 时约 7.6 MiB，属于 persistent round state。optimized route
只额外规划 3.75 MiB contiguous shortlist logits，不重复分配 full-vocabulary logits。

Workspace planner先保留跨 stage 存活的 round-live fixed regions，再对各互斥 operator
scratch stage 取峰值。它不把 prefill、target verify、DFlash proposal、decision 和
Vision/MTP-exclusive scratch 机械求和。

### 15.3 Memory summary

建议：

- DFlash weights计入 weight arena；
- active DFlash K/V和boundary K/V snapshot计入 sequence arena；
- `kv_payload_bytes` 包含 target KV、active DFlash K/V和boundary K/V snapshot，并在文档中
  明确它表示全部持久 attention cache payload；
- frontier/control计入 sequence，但不伪装成 model KV；
- feature/proposal temporaries计入 workspace。

最终字段语义需与 CLI/server memory report 同步，不能只修改内部容量。

## 16. Controller 和统计

`SpeculativeStats` 继续报告 backend-neutral：

- enabled；
- draft window；
- drafted/accepted tokens；
- speculative rounds；
- fallback steps；
- accepted per position。

建议增加 backend 字段，使结果可以区分 MTP 与 DFlash；不要通过 draft window 猜测。

Target sampling仍由现有 `SamplingConfig` 控制。DFlash proposal 固定 greedy one-hot，因此可
直接复用当前 greedy-draft stochastic accept/correction 语义。

Host controller 不需要知道 proposal model 的内部状态。它只消费：

```text
round_tokens
produced_count
pending speculative kind
```

backend-specific frontier修复由 Program 在 `resolve_pending` 中完成。

## 17. Correctness 和集成验收

本节只在全部实现phase完成、DFlash已经可以端到端产生draft并进入共享target verify后执行。
本项目没有另一条同语义DFlash engine流程可供中间tensor dump对照，因此不为各phase设置
独立正确性门槛，也不新建一套重复的integration reference engine。正确性证据由已经完成的
单Op oracle资格与最终Engine闭环共同组成；中间phase只允许编译、链接、layout/admission
检查，不能宣称DFlash集成已经正确。

### 17.1 配置与加载

- 三种 backend 的合法配置通过；
- MTP+DFlash、DFlash+Vision、27B+DFlash 均早期拒绝；
- MTP 和 DFlash 分别覆盖 full/optimized head，`none+optimized` 早期拒绝；
- real 35B artifact在每种模式只上传对应 payload；
- inactive optional view/state/graph均不存在；
- DFlash embedding/full-head aliases 和 optimized-head view 不重复占用 weight payload。

### 17.2 Feature capture

- prefill `T=1`、chunk boundary、默认 `T=1024`；
- verify `B=2..16`；
- 八个 layer slice顺序和post-MLP时点；
- Graph replay中地址和内容更新；
- 非 DFlash路径不产生feature copy；
- pitched feature pack完整写入八个slice。

### 17.3 Proposal

- 每个 `K=1..15`；
- short、full-local-window、long full-context；
- layer 0..4 SWA和layer 5 full attention；
- mask row 248077、repeated masks、same-position head；
- full Q6 head，以及 optimized Q4 shortlist/argmax/token-id remap；
- DFlash实际产生合法draft并交给target verify；
- eager/Graph在相同Engine状态下产生一致proposal和target transaction。

### 17.4 Shared transaction

对每个 `K=1..15` 和 forced `A=0..K` 检查：

```text
C = A + 1
target_frontier'  = target_frontier + C
context_frontier' = context_frontier + C
pending_anchor    = correction_or_bonus
```

检查全部 target state、六层 DFlash context、rejected candidate rows、下一轮 proposal 和统计。

### 17.5 Prefix 和回退

- exact-frontier append；
- assistant-boundary restore，回退距离分别小于和大于4096；
- local ring wrap前后保存/恢复；
- 80 MiB snapshot逐bit恢复；
- full cache只降frontier；
-连续多轮用新boundary替换旧snapshot；
- speculative batch内terminal partial，覆盖每个`R=1..C`；
- terminal exact commit后立即开始下一请求；
- changed token/media identity不能错误复用；
- DFlash Engine拒绝media request。

### 17.6 回归

- none backend ordinary output；
- MTP Text/Vision、full/optimized head；
- DFlash Text、full/optimized head；
- MTP K范围仍为1..5；
- prefix reuse和partial-terminal现有real-engine测试；
- 27B不受35B DFlash state和memory影响。

## 18. Performance 验收

增加完整 DFlash round benchmark，不能用单 Op 微基准替代：

- `K=1..15`；
- full/optimized proposal head；
- forced `A=0..K` 分离状态成本；
-真实greedy acceptance；
-短上下文、4096 local window和代表性长上下文；
- eager和CUDA Graph；
- target feature capture/context construction；
- proposal六层；
- target verify/accept；
- context append和next proposal；
- complete-round latency；
- proposal-head stage latency、每位置接受率和平均接受长度；
- accepted/published tokens per second；
-与none和MTP最佳配置对照。

full 与 optimized 必须分别按“proposal 成本 + 实际接受率 + complete-round throughput”验收。
不能因为 optimized shortlist 更快就忽略接受率，也不能因其 token 与 full-head argmax
不同而判错；最终产品默认值由端到端测量决定。

另外测量非热路径：

- 80 MiB boundary snapshot时间；
- 80 MiB restore时间；
- partial-terminal exact context flush时间；
- prefix reuse总节省是否仍显著高于snapshot/restore成本。

只有完整 round 证明组合路径没有不可接受的launch或materialization缺口后，才重新评估
deferred fusion。

## 19. 建议实现顺序

以下phase只是代码依赖顺序，不是阶段验收。`OP-A01..A07`以及`OP-N01..N04`已经全部合入并
完成各自的correctness/benchmark/route资格；集成阶段不重复这些Op工作。Phase 1..6允许做
编译、链接、静态layout和启动admission检查，但在Phase 7端到端闭环前，不以中间tensor
dump、局部状态或单独phase结果宣称DFlash正确。

### Phase 1：配置与条件加载

- public backend enum和验证；
- CLI/server统一参数；
- 35B DFlash placement；
- 27B rejection；
- immutable model view接入Program；
- MTP/DFlash/Vision load matrix接线。

### Phase 2：DFlash persistent state

- active local/full caches；
- 80 MiB boundary snapshot；
- frontier/control；
- pending context update storage；
- memory summary和reset。

### Phase 3：Target feature capture和context prefill

- production DFlash feature sink；
- workspace layout；
-八层capture；
- feature projection/norm；
-六层context K/V construction；
- boundary处snapshot；
-prefix restore后suffix重建。

### Phase 4：Eager proposal

- masked block；
- six-layer composed route；
- full head/argmax 和 optimized shortlist/argmax/remap；
- initial proposal；
-Text-only admission。

### Phase 5：完整 eager speculative transaction

- shared verify/accept复用；
- verify feature capture；
- pending context update；
- 下一轮开头的device-count exact context commit；
- current proposal；
- ordinary fallback；
- partial-terminal exact commit；
-prefix reuse和连续多轮。

### Phase 6：CUDA Graph和controller cutover

- DFlash graph ranges；
- load-time warm/capture；
- stable workspace；
- initial verify、steady round和terminal flush路径；
- capacity tail；
-统计和NVTX。

### Phase 7：产品与性能收尾

- CLI/server真实请求；
- docs和`--help`；
- load/memory reporting；
-全部Section 17最终端到端正确性与状态验收；
-完整round benchmark；
-none/MTP/DFlash回归；
-关闭Op checklist中的integration/enclosing evidence。

## 20. 第一版审阅重点

本稿请求优先确认以下决策：

1. public API使用显式`SpeculativeBackend`，不再从draft数量隐式选择MTP；
2. DFlash+Vision在Engine construction直接拒绝，而不是request-level fallback；
3. DFlash与MTP同样允许full和optimized proposal head，optimized head不隶属于MTP
   decoder；
4. 80 MiB local K/V boundary snapshot是正式prefix方案；
5. DFlash context只在host确认后exact commit，不设置per-round undo log；
6. DFlash ordinary fallback保持context frontier，并在当前请求余下部分固定走ordinary tail；
7. steady-state Graph使用“提交上一轮已确认context → proposal → verify/accept”的旋转结构。

这些决策确认后，可以从Phase 1顺序实施。全部所需Op已经合入；Phase 1..6不设置中间
正确性验收，最终证据在Phase 7的完整DFlash proposal→target verify Engine闭环中一次建立。
