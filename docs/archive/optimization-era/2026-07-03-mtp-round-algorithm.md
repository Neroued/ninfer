# MTP 每轮算法与索引约定（细分文档 1）

> Status: normative，隶属于 [总领文档](2026-07-03-mtp-spec-decode-overview.md)。
> Date: 2026-07-03。
> 本文精确定义 prefill 阶段与 decode round 的每一步：输入构造、索引推导、接受判定、
> 提交顺序、边界情形与统计口径。状态缓冲与算子状态契约见
> [state-management](2026-07-03-mtp-state-management.md)；vLLM 对应机制的源码证据见
> [vllm-reference](2026-07-03-mtp-vllm-reference.md)。

记号沿用总领文档 §2：`k`、`T_v=k+1`、`L`、`t0`、`d1..dk`、`a`、`t*`。
所有 shape 按 qus ne 序（fastest dim first），`[5120,T]` 表示 T 列 hidden。

---

## 1. Prefill 阶段

### 1.1 target chunked prefill（现有路径 + 一个新输出）

对 prompt `x_0..x_{P-1}` 按现有 chunk 循环处理。唯一新增要求：每个 chunk 在
64 层结束后，对**整个 chunk** 计算 final RMSNorm，产出
`H_c = hidden[5120, n_c]`（现状只对最后 chunk 的最后一列计算）。`H_c` 写入
chunk 级 hidden 缓冲，供本 chunk 的 MTP shifted pass 消费，随后可被下一 chunk
覆盖。logits/argmax 仍只在最后 chunk 的最后一列计算，得到 `t0`。

### 1.2 MTP shifted pass（每 chunk 一次，紧跟 target chunk）

对起点 `c0`、长度 `n` 的 chunk：

```text
mtp_ids[j]   = x_{c0+j+1}            j ∈ [0, n-2]
mtp_ids[n-1] = x_{c0+n}              非最后 chunk（下一 chunk 的首 token，"next_prefill_token"）
             = t0                    最后 chunk（target 刚采样的 token）
positions    = c0 .. c0+n-1          （照抄 target chunk 的 positions）
hidden 输入  = H_c[:, 0..n-1]
```

计算流程（shape 流水见 Part 2 §11）：

```text
e = rmsnorm(embed(mtp_ids), pre_fc_norm_embedding, 1+w)     # [5120,n]
h = rmsnorm(H_c,            pre_fc_norm_hidden,    1+w)     # [5120,n]
x = mtp.fc(concat(e, h))                                    # [10240,n] -> [5120,n]
x = mtp_layer(x, positions)      # 1 层 full-attn：append-batch attention 写 MTP KV @ c0..c0+n-1
m = rmsnorm(x, mtp.norm, 1+w)                               # [5120,n]
```

- 每行的语义：`(target hidden at 位置 p, token at 位置 p+1)` → 预测位置 `p+2`。
- **非最后 chunk 不算 logits**（省一次 lm_head 读），只为填 MTP KV。
- 最后 chunk 额外取最后一行 `m[:, n-1]`，`d1 = argmax(lm_head(m[:, n-1]))`。
- MTP KV 采用 fill-then-attend：先写本窗口 K/V 再做因果 attention，行 `j` 可见
  位置 `0..c0+j`。
- `P == 1` 的退化情形：`mtp_ids = [t0]`，hidden = `H[:,0]`，仍然成立。

### 1.3 prefill 收尾的 AR steps

最后 chunk 的 shifted pass 给出 `d1` 与 `m[:, n-1]`（记作 MTP AR hidden）。随后
执行 `k-1` 次 AR step 得到 `d2..dk`，与 decode round 的 AR step（§2.4）完全同构：
首个 AR step 的输入为 `d1`，`ar_pos = P`（= 最后一行位置 `P-1` 加一）。
prefill 结束时的全局状态：

```text
L = P;  t0 = argmax(target logits);  drafts = d1..dk
target KV/GDN 覆盖 0..P-1；`gdn_initial_slot=0`；MTP KV 覆盖 0..P-1
（另有 AR step 写入的 P..P+k-2 draft 条目）
```

---

## 2. Decode round

每轮四个阶段严格按序：verify → accept/selector-update → propose(shifted) → propose(AR)。
所有 kernel 形状只依赖 `k`；`a` 只以 device 标量形式参与索引。

### 2.1 verify

输入构造（device kernel 完成，无 host 参与）：

```text
verify_ids[0]   = t0            # 来自 io_.token
verify_ids[1+i] = drafts[i]     # i ∈ [0,k)
positions[j]    = L + j         # j ∈ [0,k]，写入 positions 缓冲
```

target forward，`T = T_v = k+1`：

- 16 个 full-attn 层：append-batch attention，`cache_offset = L`（device 标量），
  物理写 target KV @ `L..L+k`，行 `j` 的注意力窗口为 `0..L+j`（fill-then-attend +
  窗口内因果 mask）。
- 48 个 GDN 层：conv sequence + gated delta recurrent 的 **snapshot 变体**：
  `state_in = slot[gdn_initial_slot]`，逐 token 处理并把 after-token-j 状态写入
  `slot[j]`（详见 state-management §4/§5）。数学与现有逐 token decode 完全一致
  （同一 recurrent kernel 的串行 token 循环）。
- MLP/norm 与 prefill 同构，T 维放宽到 `k+1`。
- 末端对**全部 k+1 列**计算 final RMSNorm → `hidden[5120,k+1]`（写入 verify
  hidden 缓冲），lm_head → `logits[248320,k+1]`，逐列 argmax →
  `target_tokens[k+1]`，其中 `target_tokens[i] = g_i`。

### 2.2 accept + selector update

单个小 kernel（"accept kernel"）完成判定与标量更新：

```text
a = 0
while a < k and target_tokens[a] == drafts[a]:   a += 1
t* = target_tokens[a]

sampled_out[i] = drafts[i]      i < a
sampled_out[a] = t*
num_sampled    = a + 1
io_.token      = t*                    # 下一轮 t0
L_next         = L + a + 1             # 更新 committed length 标量
```

标量更新时序说明：本节及 §2.3/§2.4 文字中的 `L` 一律指**轮初值**。为避免
propose 读到已更新的值，约定两个 device 标量：`window_base`（verify 输入构造时
从 `L` 拷贝，整轮不变，作为 verify 与 shifted pass 的窗口基址）与 `L`
（committed length，accept kernel 内更新为 `L+a+1`）。shifted pass 的 positions
直接复用 verify 填好的 positions 缓冲；`ar_pos` 以更新后的 `L` 初始化
（数值上 = 轮初 `L+a+1`）。

提交动作（`a` 只经 device 标量传播）：

1. **target KV**：逻辑回退 `pos := L+a+1`。物理数据不动；下一轮窗口
   `L'..L'+k` 覆盖 stale 条目。attention 算子的窗口边界一律由 device 位置标量
   推导，host 侧 `kv_.pos` 仅在每轮回读后同步，用于容量检查。
2. **GDN**：device 小 kernel 写 `gdn_initial_slot := a`。下一次 verify 或
   MTP-enabled fallback decode 由 conv/ssm kernel 自行按 selector 读取初始槽位。
3. **MTP KV**：逻辑回退到 `L+a+1`（与 target KV 同语义；物理覆盖由下一轮
   shifted pass 保证，见 §2.3 与总领 C4）。
4. 统计计数器累加（§4）。

顺序约束：selector update 必须在本轮 GDN snapshot 写全部完成后、下一次 GDN 读之前
执行；放在 accept kernel 之后、propose 之前即可（propose 不触碰 GDN）。

### 2.3 propose — shifted pass（固定 T = k+1）

输入构造（device kernel）：

```text
shifted_ids[j] = verify_ids[j+1]       j ∈ [0,k)     # 左移一位（即 d1..dk）
shifted_ids[a] = t*                                   # 用 device 标量行号覆盖
# 行 a+1..k 为 junk，不修正
positions      = 复用 verify 的 positions 缓冲（L..L+k）
hidden 输入    = verify hidden[:, 0..k]（前 a+1 列有效）
```

MTP forward 与 §1.2 相同（T = k+1），append-batch 写 MTP KV @ `L..L+k`。
第一个新 draft：

```text
gather 行 a：m_a = mtp_hidden[:, a]        # device 标量行号 gather
d1' = argmax(lm_head(m_a));  drafts[0] = d1';  ar_hidden = m_a
ar_pos = L + a + 1
```

junk 行（`a+1..k`）照常计算并写 KV：位置 `L+a+1..L+k` 的条目是垃圾，但

- 本 pass 内因果 mask 保证行 `≤ a` 读不到它们；
- 本轮 AR step（写 `L+a+1..`）与下一轮 shifted pass（写 `L'..L'+k`，
  `L' = L+a+1`）会将其覆盖；上一轮全部 stale 条目位置 `< L+k`，同理被本
  pass 覆盖（fill-then-attend 使覆盖先于读取）。

### 2.4 propose — AR steps（k-1 次，T = 1）

对 `i = 1..k-1`：

```text
e = rmsnorm(embed(drafts[i-1]), pre_fc_norm_embedding, 1+w)   # [5120,1]
h = rmsnorm(ar_hidden,          pre_fc_norm_hidden,    1+w)
x = mtp.fc(concat(e, h))
x = mtp_layer(x, position = ar_pos)     # decode 式 attention，窗口 0..ar_pos，写 MTP KV @ ar_pos
m = rmsnorm(x, mtp.norm, 1+w)
drafts[i] = argmax(lm_head(m));  ar_hidden = m;  ar_pos += 1
```

- AR step 的注意力前缀 = committed 前缀 + 本轮已接受行 + 已产 draft 行，正是
  draft 视角所需的上下文。
- `k == 1` 时无 AR step，propose 只有 shifted pass。

### 2.5 host 收尾（每轮一次）

回读 `sampled_out[0..k]` 与 `num_sampled`（一次小 D2H，同今日 `read_token` 模式），
然后：

1. 依次向输出流提交 `sampled_out[0..num_sampled-1]`；
2. 遇 stop token：输出在该 token 处截断，本轮剩余 token 丢弃，结束生成；
3. 达到 `max_new_tokens`：**overshoot-and-truncate** —— 轮内多产出的 token 直接
   不进入输出（状态里已提交，但会话即将结束，无影响）；
4. 同步 host 侧位置镜像（`kv_.pos`、`mtp_kv.pos`、`L`）。

---

## 3. 回退与容量边界

1. **容量守卫**：仅当 `L + 2k ≤ max_ctx` 时运行 MTP round（轮内最远触及位置
   `max(L+k, L+a+k-1) ≤ L+2k-1`，再留下一轮窗口余量由下轮自身判断）。否则回退
   到 T=1 decode 步直至结束。若 MTP 已启用，回退 decode 的 GDN conv/ssm 从
   `gdn_initial_slot` 读取初始状态，写 after-token 状态到 slot 0，然后重置 selector。
2. **禁止事项**：round 循环内不得调用 `Engine::prefill`（它会 reset KV/GDN）；
   不得在轮中途以 host 分支改变 kernel 序列形状（违反总领 C5）。

---

## 4. 统计口径

与 spec-decode 常见吞吐指标对齐：

| 指标 | 定义 |
|---|---|
| `draft_tokens` | 累计送验 draft 数 = k × 轮数 |
| `accepted_tokens` | 累计 `a` |
| `acceptance_rate` | `accepted_tokens / draft_tokens` |
| `acceptance_length` | 每轮平均产出 = `1 + accepted_tokens / 轮数` |
| `accepted_per_pos[i]` | 位置 i 的 draft 被接受次数（i ∈ [0,k)，衡量位置衰减） |
| `rounds` / `fallback_steps` | MTP 轮数 / 回退普通 decode 步数 |

计数在 accept kernel 内用 device 计数器累加，随 bench 报告输出（schema 变更见
implementation-requirements §7）。

---

## 5. 正确性论证要点

1. **接受即自洽**：committed 序列的每个 token 要么是 target argmax 的直接输出
   （`t*`、prefill 的 `t0`），要么在接受判定中与 target argmax 相等（`d_i`）。
   因此 committed 序列恒等于 verify 路径下 target 的 greedy 轨迹。
2. **状态一致**：C2/C3/C4 保证三类状态在每轮结束时都精确对应 committed 前缀
   `0..L'-1`：KV 靠位置轴 + 覆盖，GDN 靠 snapshot 选择，MTP KV 靠位置轴 + 覆盖。
3. **kernel 路径差异**：batched verify 与 T=1 decode 的浮点归约顺序不同，
   near-tie 处 argmax 可能与普通 decode 分叉。分叉后的序列仍满足本文不变量
   1/2（自洽）；MTP 验收不以跨实现 token 序列相等为目标。
4. **MTP 质量无关性**：drafts 只出现在接受判定的右侧与 MTP 自身输入中，任何
   draft 错误只减小 `a`，不进入 committed 状态。

---

## 6. 与 vLLM 的逐点对应

| 本文步骤 | vLLM 对应（详见 vllm-reference） |
|---|---|
| §2.1 verify_ids 组装 | `combine_sampled_and_draft_tokens`（input_batch.py） |
| §2.1 全窗口 logits | `logits_indices` = 每请求窗口末 `num_logits` 行 |
| §2.2 accept | rejection kernel `temp==0` 分支 + `_insert_resampled`（bonus/replacement） |
| §2.2 KV 逻辑回退 | `num_computed_tokens += query_len - num_rejected` |
| §2.2 GDN selector | 下一步 kernel 按 `num_accepted_tokens` 选 slot |
| §2.3 shifted pass | `prepare_prefill_inputs`（shift-left、last_sampled、positions 照抄、junk padding） |
| §2.4 AR steps | `prepare_decode_inputs` / `update_draft_inputs` / `_multi_step_decode` |
| §1.2 chunk 级 shifted pass | chunked prefill 时 drafter 用 `next_prefill_tokens` 收尾 |
| §3.1 容量守卫 | scheduler 的 max-model-len 裁剪 |

qus 的简化：非最后 chunk 跳过 MTP logits 与 AR steps（vLLM 因 batching 无法跳，
靠 scheduler 丢弃提案）；k 固定不裁剪（vLLM 会按 budget 裁剪 draft 数）。

---

## 7. 数值细节备忘

1. MTP 全部 RMSNorm 为 `(1+w)` 约定（同 target 层规范）；MTP 无 GDN gated norm。
2. RoPE：MTP 层与 target full-attn 层同参数（partial 64 dims、theta 1e7、
   text-only 下 MRoPE 退化为 1D）。positions 为绝对逻辑位置。
3. `attn_in.w8` 输出行区间 `q[0,6144) k[6144,7168) gate[7168,13312) v[13312,14336)`；
   `T>1` 时需 compact split（Part 2 §7 Option A/B）。
4. argmax tie-break：全项目统一"最低 index 获胜"（现有 `kernels::argmax` 行为），
   verify 与 draft 两侧一致。
5. lm_head/embedding 为 target 与 MTP 共享张量（checkpoint 无独立 MTP 副本）。

---

## 8. 概率采样扩展（预留，非 v1）

未来接入真实 sampler 后，本轮结构不变，仅替换判定与采样规则（vLLM 约定，
证据见 vllm-reference §4）：

1. draft 侧按 temperature-only 采样并保存 processed draft logits
   `q_i`（`[k, vocab]` fp32 缓冲）；top-k/p 不作用于 draft。
2. target 侧对窗口 logits 施加完整采样管线（temperature/top-k/top-p/惩罚）得
   `p_i`；接受判定 `u < min(1, p_i(d_{i+1})/q_i(d_{i+1}))`。
3. 拒绝位从残差分布 `max(p-q, 0)` 重采样；全接受时 bonus 从 `p_k` 采样；
   draft 为 greedy 时 `q` 视作 one-hot（残差 = 屏蔽被拒 token 的 `p`）。
4. 随机数按 `(seed, 绝对位置)` 键控；draft 采样用 `position+1` 使同一逻辑槽位
   的噪声在 draft/verify 两侧一致，从而输出与接受模式无关、可复现。

在此之前，qus 的 greedy 约定是上述规则在 `temperature → 0`、`q` one-hot 下的
精确特例，未来切换不改变轮结构与状态管理。
