# MTP Speculative Decoding 总体约定（总领文档）

> Status: normative（本文是 qus MTP/speculative decoding 的顶层约定，冲突时以本文为准）。
> Date: 2026-07-03。
> Scope: Qwen3.6-27B 自带 MTP head 的 self-speculative decoding，batch=1，单 RTX 5090。
> 证据基线: 本地 vLLM 树 `~/vllm`（commit `92221485aaaa`, 2026-06-25）；本仓库 commit `bd1d149`。

本文定义整个项目的 MTP 算法约定：术语、角色分工、每轮 decode 的状态机、五条核心
convention、状态所有权、内存与性能模型、正确性契约，以及文档层级。具体算法步骤、
状态管理细节、vLLM 源码证据、实现需求分别在四篇细分文档中展开（见 §9）。

---

## 1. 目标与非目标

### 1.1 目标

1. 用 checkpoint 自带的 MTP head 做 self-speculative decoding，提高 batch=1 decode
   吞吐（headline metric）。
2. 算法层面与 vLLM v1 的 MTP 路径对齐：同样的 verify/propose 轮结构、同样的
   draft 输入构造、同样的状态提交语义。工程层面按 batch=1 静态调度专门化。
3. 保持 target 模型的权威性：MTP 只影响吞吐，不改变（greedy 语义下的）输出正确性。

### 1.2 非目标（v1 明确不做）

1. 概率采样 + rejection sampling。当前引擎 sampler 只有 greedy argmax，v1 的验收
   规则固定为 greedy token-equality。概率化扩展的算法约定在 round-algorithm 文档
   §8 预留，vLLM 的完整机制在 vllm-reference 文档 §4 记录，但不属于 v1 实现范围。
2. 多请求调度、continuous batching、paged KV/block table、prefix caching。
3. Tree/多分支 speculation（如 EAGLE tree attention）。每轮只有一条 draft 链。
4. 独立 draft model。draft 权重固定为本 checkpoint 的 `mtp.*` 张量（q5090 v3
   MTP_DRAFT module，W8G32）。
5. structured output / grammar 约束、per-request 动态开关 MTP。

---

## 2. 术语与记号（全套文档共用）

| 记号 | 含义 |
|---|---|
| `k` | 每轮 speculative draft 数，engine load 时固定，`k ∈ [1,5]` |
| `T_v` | target 验证窗口 token 数，恒等于 `k+1` |
| `L` | committed length：target 状态已提交的 token 数（位置 `0..L-1` 已折入 KV/GDN） |
| `t0` | pending token：上一轮采样/接受产出、但尚未经过 target forward 的下一个输入 token，逻辑位置 `L` |
| `d1..dk` | 本轮待验证的 draft tokens，逻辑位置 `L+1..L+k` |
| `a` | 本轮接受的 draft 数，`a ∈ [0,k]` |
| `t*` | 本轮 target 产出的新 token：`a<k` 时是 replacement（拒绝位的 target argmax），`a==k` 时是 bonus |
| `num_sampled` | 本轮进入输出序列的 token 数，恒等于 `a+1`（`d1..da` + `t*`） |
| committed state | 与 committed length 一致的模型状态（target KV 前缀、GDN conv/ssm、MTP KV 前缀） |
| shifted pass | MTP 的"prefill 式"批量 forward：ids 左移一位 + 末尾补 next token，hidden 用 target 同窗口 hidden |
| AR step | MTP 的自回归 draft 步：输入上一 draft token + 上一 MTP hidden，T=1 |
| round | 一次完整的 verify → accept → propose 循环 |

位置恒等式（每轮成立）：

```text
round 开始:  target 状态 = 前缀 0..L-1;  t0 在位置 L;  drafts 在 L+1..L+k
verify:      target 一次 forward T_v = k+1 个 token, 物理写位置 L..L+k
accept:      得到 a 与 t*;  输出 d1..da, t*
round 结束:  L' = L + a + 1;  t0' = t*;  新 drafts d1'..dk' 在 L'+1..L'+k
```

---

## 3. 角色分工（核心 mental model）

```text
target model (64 层):   权威 verifier + 状态 owner。
                        照常 prefill/decode/verify，产生 hidden、logits、argmax。
MTP head (1 层 full-attn): 轻量 learned proposer。
                        读 target hidden 与 token ids，产 draft tokens。
                        拥有独立 MTP KV，不读写 target 任何状态。
engine/runtime:         orchestrator。
                        持有全部状态与 device 标量，驱动 round 状态机，
                        执行 accept/commit/rewind，向上层交付 token 流。
```

与 vLLM 一致的边界：MTP 与主流程的交互点只有 hidden states、token ids、
positions、draft tokens（v1 无 draft logits）。MTP 不修改 target 层状态，draft
token 未经 target 验证不进入 committed 上下文。

---

## 4. 全生命周期状态机

```text
[prefill 阶段]
  for each prompt chunk c:
    target chunked prefill(c)            # 现有路径; 额外产出全 chunk 的 final-norm hidden
    mtp shifted pass(c)                  # 填充 MTP KV; 非末 chunk 用 next_prefill_token 收尾
  最后 chunk: target 采样 t0 (greedy argmax)
  mtp shifted pass(last chunk, 末位补 t0) → 产出 d1
  mtp AR steps × (k-1) → d2..dk
  L = prompt_len

[decode 轮循环]  while 未停止:
  verify:   target forward [t0, d1..dk] @ 位置 L..L+k
            → hidden[5120, k+1], logits[vocab, k+1], target_tokens[k+1] (per-column argmax)
  accept:   a = 最长匹配前缀; t* = target_tokens[a]; 输出 d1..da, t*
  commit:   L += a+1; target KV pos := L; GDN commit-copy(slot a); MTP KV pos 同步
  propose:  mtp shifted pass(固定 T=k+1, 有效长 a+1, 末位补 t*) → d1'
            mtp AR steps × (k-1) → d2'..dk'
  host:     回读本轮 sampled tokens (a+1 个), 检查 stop/上限

[收尾]
  剩余 token 预算 < 2 或接近 max_ctx 时，回退普通 T=1 decode 步
```

细化的索引推导、边界情形与伪代码见 round-algorithm 文档。

---

## 5. 五条核心 convention（normative）

### C1 — 验证窗口与 greedy 接受规则

每轮 target 恰好 forward `T_v = k+1` 个 token：`[t0, d1..dk]`，并对全部 `k+1` 个
位置取 logits 与 argmax（记 `g_i = argmax(logits[:,i])`）。接受规则：

```text
a = max { j ∈ [0,k] : ∀ i < j, g_i == d_{i+1} }
t* = g_a
本轮输出 = d1..da, t*    (num_sampled = a+1)
```

- `t0` 天然属于 committed 序列（它是上一轮的输出），不参与接受判定。
- `a == k` 时 `t*` 即 bonus token：所有 draft 被接受后仍多得一个 token。
- 每轮输出 token 数在 `[1, k+1]`；MTP 全错时退化为普通 decode 的产出速率。
- 该规则与 vLLM greedy 路径（rejection kernel 的 `temp==0` 分支）逐字对应，也与
  llama.cpp 的 sample-and-match 一致。

### C2 — committed length 记账 + KV「逻辑回退、物理覆盖」

`L` 是唯一权威的序列进度。target KV 与 MTP KV 都带位置轴，拒绝的 suffix 不清理：

```text
verify 物理写 target KV @ L..L+k
commit 时 kv.pos := L+a+1   (逻辑回退)
下一轮 verify 写 L'..L'+k，天然覆盖上一轮的 stale 条目
```

attention 读取窗口由 `pos`/长度约束，stale 条目永远不可见。这与 vLLM 完全一致
（`num_computed_tokens += query_len - num_rejected`，slot 由位置推导，下一步覆盖）。
`KVCache` 需要新增回退能力（设置 `pos` 到指定值），语义上只是移动游标。

### C3 — GDN 状态：per-token snapshot + commit-copy（deferred commit）

48 个 GDN 层的 conv/ssm 状态没有位置轴，一旦折入 token 无法逻辑回退。约定采用
vLLM 的 deferred-commit 思想、按 batch=1 简化：

```text
每 GDN 层的状态缓冲扩展为 slot[0..k]（ssm 每槽 [128,128,48] fp32，conv 每槽 [10240,3] bf16）
slot[0] 就是 committed 状态；prefill/普通 decode 路径只读写 slot[0]，行为不变
verify 阶段的 conv/recurrent 算子: 以 slot[0] 为初始状态，
    逐 token 处理 [t0, d1..dk]，处理完第 j 个 token 后把状态快照写入 slot[j]
commit: 由 device 标量 a 驱动一次 gather-copy: slot[a] → slot[0]（a==0 时为 no-op）
```

要点：
- token 0（`t0`）永远属于 committed 序列，所以用 after-t0 快照覆盖 slot[0] 是
  安全的（初始状态先读入寄存器再写，无冒险）；round 之间 slot[0] 恒等于
  committed 状态。
- 不做「快照-回滚-重放」：重放需要重跑整轮 forward 或按层缓存中间量并二次发射
  kernel，数据依赖使 kernel 序列不固定，无法进 CUDA graph。snapshot 方案的
  kernel DAG 与 `a` 无关（见 C5）。
- 与 vLLM 的差异只在 commit 形态：vLLM 用 `num_accepted_tokens` 在下一步做
  indirect read（省一次拷贝），qus v1 用显式 commit-copy 换取「round 之间
  committed 状态唯一且可 dump」的强不变量；indirect-read 作为性能后续项。
- chunked prefill 与普通 decode 路径不变，仍就地读写 slot[0]。

### C4 — MTP KV 独立命名空间 + 固定形状 propose（junk 覆盖）

MTP 层拥有独立的 1 层 KV cache，位置轴与 target 逻辑位置对齐。propose 的
shifted pass **固定跑满 `T = k+1` 行**（与 verify 窗口同形状），其中只有前
`a+1` 行有效；无效行照常写 KV，成为 junk：

```text
shifted ids:   [x_{L+1}, ..., x_{L+a}, t*] + (k-a) 行 junk       # ids 左移一位、末位补 t*
hidden 输入:   verify 产出的 hidden[:, 0..k]（前 a+1 列有效）
positions:     L..L+k（照抄 verify 窗口）
第一个 draft:  d1' = argmax(mtp_logits[:, a])                    # 行号 a 由 device 标量给出
AR step j:     输入 d_j'，位置 L+a+j，注意力窗口 0..L+a+j
```

junk 行写入的 MTP KV（位置 > L+a 的部分）会被本轮 AR steps 以及下一轮 shifted
pass 按位置覆盖，因果 mask 保证有效行读不到 junk。MTP KV 在位置 `p` 的语义恒为
「(token `p+1`, target/MTP hidden at `p`) 的编码」——即整体右错一位的 draft 视角。
这与 vLLM 的 padded-drafter-batch 机制逐点对应。

### C5 — 整轮固定形状、device 标量驱动（CUDA-graph 友好）

一轮 round 的全部 kernel 形状只依赖常量 `k`，不依赖 `a`。数据依赖全部通过
device 标量/小缓冲传递：

```text
device 标量/缓冲: L (io_.pos 扩展), window_base (轮初 L 快照), a, valid_len = a+1,
                  t0 (io_.token 扩展), draft_tokens[k], sampled_out[k+1],
                  verify_ids[k+1], shifted_ids[k+1]
host 每轮只做:    回读 sampled_out 与 num_sampled，做 stop/长度判断
```

由此整轮（verify + accept + commit + propose）是一张固定 kernel DAG，可以像现有
decode graph 一样捕获成单个 CUDA graph。这是 vLLM「avoid CPU-GPU sync +
uniform batch」设计在 batch=1 下的直接翻译。v1 允许先 eager 跑通，但**算子与
数据流契约必须从一开始就满足本条**（不得引入依赖 `a` 的 host 分支/形状）。

---

## 6. 状态所有权总表

| 状态 | owner | 形状/精度 | 生命周期与提交语义 |
|---|---|---|---|
| target full-attn KV ×16 | Engine（现有 `KVCache`） | `[256, padded_ctx, 4]` bf16 ×2 | 物理 append @ L..L+k；commit 时 `pos := L+a+1`；stale 覆盖 |
| target GDN conv ×48 | Engine（`GdnState` 扩展） | slot[0..k] × `[10240,3]` bf16 | verify 逐 token 快照；commit-copy slot[a]→slot[0] |
| target GDN ssm ×48 | Engine（`GdnState` 扩展） | slot[0..k] × `[128,128,48]` fp32 | 同上 |
| target hidden (verify 窗口) | Engine 新缓冲 | `[5120, k+1]` bf16 | 每轮覆盖；MTP shifted pass 的 hidden 输入 |
| target hidden (prefill chunk) | Engine 新缓冲 | `[5120, prefill_chunk]` bf16 | 每 chunk 覆盖；chunk 级 MTP shifted pass 输入 |
| MTP full-attn KV ×1 | Engine 新增第二个 `KVCache` 实例 | `[256, padded_ctx, 4]` bf16 ×2 | 与 target KV 同样的逻辑回退/物理覆盖 |
| MTP AR hidden | Engine 新缓冲 | `[5120, 1]` bf16 | AR step 之间传递上一 MTP hidden |
| draft tokens | Engine device 缓冲 | `[k]` i32 | propose 写，下一轮 verify 读 |
| verify logits | Engine（`StepState` 扩展） | `[248320, k+1]` bf16 | 每轮覆盖 |
| 位置/接受标量 | Engine device 标量 | `L, a, t0` 等 | C5 所列 |

MTP 不使用任何 GDN 状态（MTP 层是纯 full attention）；MTP 不写 target KV；
target 不读 MTP KV。

---

## 7. 内存与性能模型

### 7.1 增量内存（k=4 / k=5，max_ctx=8192）

| 项 | 公式 | k=4 | k=5 |
|---|---|---|---|
| GDN ssm snapshot（新增 k 槽） | `48 × k × 3.0 MiB` | 576 MiB | 720 MiB |
| GDN conv snapshot（新增 k 槽） | `48 × k × 60 KiB` | 11 MiB | 14 MiB |
| MTP KV（1 层, 8192 ctx） | `2 × 256×8192×4×2B` | 32 MiB | 32 MiB |
| MTP 权重（W8G32 payload） | 固定 | 431 MiB | 431 MiB |
| verify logits/hidden/杂项 | — | < 20 MiB | < 20 MiB |
| 合计 | — | ≈ 1.07 GiB | ≈ 1.22 GiB |

ssm snapshot 是大头，且随 k 线性增长。这是 deferred-commit 的固有代价，与 vLLM
的 `num_speculative_blocks` 机制同源；在 32 GB 卡上可接受，但 `k` 必须是 load
时常量以便 arena 预算。

### 7.2 吞吐模型（roofline 级）

记 target 一步 decode 成本 `C_t`（权重带宽主导，当前实测 ~52 tok/s）。因为
weight-bound GEMV 在 `T ≤ 6` 时权重流量不变，verify 一轮成本 ≈ `C_t × (1+ε)`
（ε 为 small-T 效率损失，未调优前可能到 0.2~0.5）。MTP 侧每次产 logits 要读
lm_head（Q6 ≈ 1.0 GB）+ MTP 权重（0.43 GB），k 步 draft 的 MTP 成本
≈ `k × 0.8 ms` 量级。每轮产出 `E[a]+1` 个 token：

```text
speedup ≈ (E[a]+1) × C_t / (C_t×(1+ε) + k × C_mtp)
```

用早期离线 greedy 估计数据（代码类 prompt）：k=4 时 acceptance length
约 4.5，k=5 时约 4.85。代入 ε=0.2、C_mtp≈0.8ms、C_t≈19ms：k=4 → ≈3.7×，k=5 → ≈3.8×。
保守预期（真实负载接受率更低、small-T 未调优）为 **2~3.5×**。`k` 的最终默认值
由 benchmark 决定，初始建议 3 或 4。

M3 checkpoint（见 `docs/bench/mtp-m3-target-verify-cost.md`）在 RTX 5090 上测得
generic target verify 的 k=5/T=6 成本为 `66.4134 ms`，T=1 decode 为 `13.1765 ms`，
即 ε≈4.04，远高于上面的乐观占位值。M5 必须优先处理 target small-T verify
kernel/launch 开销，否则端到端加速模型会被 verify 成本主导。

注意：acceptance 数据目前只有单 prompt 证据，正式结论以 bench 报告为准。

---

## 8. 正确性契约与验证策略

### 8.1 契约

1. **自洽性（硬性）**：committed 输出必须恰好是 verify 路径自身的 greedy 输出——
   即被接受的 `d_i` 必须等于 verify 计算出的 `g_{i-1}`，`t*` 必须是 `g_a`。这由
   C1 的构造保证，是算法层不变量。
2. **跨路径输出不作为验收目标**：batched verify（T=2..6）与普通 T=1 decode 走不同
   kernel（GEMV multistep vs T1、attention append vs split-KV decode），浮点归约
   顺序不同，argmax 在 near-tie 处可能翻转并引起序列分叉。MTP 验收只要求自身
   committed 序列满足契约 1，并正确处理容量、stop、`max_new` 与统计。
3. **接受率只影响吞吐**：MTP 质量差只降低 `E[a]`，不得影响契约 1。

### 8.2 验证策略

1. **算子级**：small-T 算子继续按现有 l1-op-test-standard 容差体系验证数学性质；
   GDN sequence 算子验证快照与 committed slot 选择语义。
2. **端到端**：canonical fixtures 覆盖 MTP batched round 能生成请求 token 数、记录
   round/acceptance/fallback 统计，并正确处理容量 fallback、stop token 截断和
   `max_new` overshoot。
3. **报告**：acceptance rate / acceptance length / tokens-per-round 进入 bench
   报告 schema，回归监控。

---

## 9. 文档层级与权威关系

```text
本文（总领, normative）
├── 2026-07-03-mtp-round-algorithm.md          # 细分: 每轮算法与索引推导, prefill 交互, 边界情形
├── 2026-07-03-mtp-state-management.md         # 细分: KV/GDN/MTP-KV 状态管理与算子状态契约
├── 2026-07-03-mtp-vllm-reference.md           # 细分(descriptive): vLLM 机制与源码证据, 对齐/偏离清单
├── 2026-07-03-mtp-implementation-requirements.md  # 实现需求: 分层 gap 清单与验证要求
└── 2026-07-03-mtp-roadmap.md                  # 落地路线: 里程碑顺序/边界/gate（M0..M5）
        └── docs/plans/ 下的正式实现计划（每个里程碑一个, 按 AGENTS.md 计划规范另立）
```

既有文档的定位（均降为参考，与本套冲突时以本套为准）：

| 文档 | 定位 |
|---|---|
| `qwen3.6-27b-mtp.md` | MTP head 结构与原理背景，仍然准确；算法约定以本套为准 |
| `2026-07-02-mtp-foundation-part1-design.md` + `2026-07-03-mtp-foundation-part1-verification.md` | 已执行的权重/ref-model 准备工作，事实记录 |
| `2026-07-02-mtp-foundation-part2-operators.md` | 算子缺口分析仍有效；**§5 的 W8G128 参数已过时（最终决定为 W8G32）**；其 GDN `state_in/state_out` 契约被 state-management 文档 §5 的 per-token snapshot 契约取代 |
| `q5090_packed_file_format_v3.md` | MTP_DRAFT 权重布局的 normative 来源，不受本套影响 |

## 10. 与 vLLM 的对齐/偏离摘要

| 主题 | vLLM | qus 约定 | 性质 |
|---|---|---|---|
| verify 窗口 `[t0, d1..dk]`、k+1 个 logits | ✓ | 同 | 对齐 |
| greedy 接受 = argmax 相等，t* = 拒绝位 argmax / bonus | ✓ | 同 | 对齐 |
| KV 逻辑回退 + 物理覆盖 | ✓ | 同 | 对齐 |
| GDN per-token 状态快照、按 accepted 选择 | ✓（k+1 slot + 下一步 indirect read） | 同思想；slot[0] 即 committed + 显式 commit-copy | 简化 |
| conv 状态 | 加宽窗口 `w-1+k` + offset 读 | 与 ssm 同构的 per-token snapshot 槽 | 简化（等价语义） |
| MTP shifted pass：ids 左移、positions 照抄、末位补 last_sampled/next_prefill | ✓ | 同 | 对齐 |
| propose 固定形状、junk 行覆盖 | ✓（padded drafter batch） | 同（batch=1 版本） | 对齐 |
| chunked prefill 每 chunk 跑 MTP shifted pass 填 KV | ✓ | 同；但非末 chunk 跳过 AR steps | 简化 |
| draft 采样 | greedy argmax（默认）/ probabilistic | greedy argmax | 对齐（子集） |
| rejection sampling / draft_logits / gumbel 种子约定 | ✓ | v1 不做，约定预留 | 延后 |
| 多请求、block table、prefix caching、grammar | ✓ | 不做 | 范围外 |
