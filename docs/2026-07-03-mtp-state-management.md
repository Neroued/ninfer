# MTP 状态管理与算子状态契约（细分文档 2）

> Status: normative，隶属于 [总领文档](2026-07-03-mtp-spec-decode-overview.md)。
> Date: 2026-07-03。
> 本文定义 speculative decoding 下三类持久状态（target KV、target GDN、MTP KV）
> 的缓冲布局、提交/回退语义、算子级状态契约，以及内存与确定性要求。
> 每轮各阶段何时读写这些状态见 [round-algorithm](2026-07-03-mtp-round-algorithm.md)。
> 本文 §5 的算子契约**取代** Part 2 §9.2 的 `state_in/state_out` 单终态契约。

---

## 1. 现状回顾（约定的出发点）

| 状态 | 现有实现 | 对 spec decode 的缺口 |
|---|---|---|
| target KV ×16 | `KVCache`：`k/v [256, padded_ctx, 4]` bf16；`pos` 单调 `advance()`，无回退 | 需要 `rewind(pos)`；需要 device 标量驱动的窗口写入 |
| target GDN ×48 | `GdnState`：`conv [10240,3]` bf16、`ssm [128,128,48]` fp32；所有算子就地无条件提交整段 T | 无快照；verify 拒绝会污染 committed 状态 |
| MTP KV | 不存在 | 需要独立 1 层 KV 命名空间 |
| hidden/logits | `StepState.logits [vocab,1]`；prefill 只算最后一列 final norm | verify 需全窗口 hidden/logits；prefill 需全 chunk hidden |

---

## 2. Target full-attn KV：逻辑回退、物理覆盖

### 2.1 语义

1. verify 阶段物理写位置 `L..L+k`（16 层全部）。
2. commit 时逻辑长度回退到 `L+a+1`。数据不清理；`pos` 之后的条目定义为 dead。
3. 下一轮窗口从 `L' = L+a+1` 起，重写 `L'..L'+k`，天然覆盖全部 dead 条目
   （上一轮触及的最远位置 `L+k = L'+k-a-1 < L'+k`）。
4. attention 读取范围由窗口推导（行 `j` 见 `0..base+j`），dead 条目永不可读。

### 2.2 实现要求

- `KVCache` 增加回退能力：把 `pos` 设为不大于当前值的目标值（语义上仅移动游标；
  现有 `slot()` 的 `position >= pos` 守卫自动使 stale 区间不可读）。
- verify/propose 的 attention 算子从 **device 标量** 取窗口基址（现有
  `gqa_attention_decode` 已从 device 读 `pos`，prefill 版本用 host `cache_offset`
  —— graph 化需要 device 标量变体，见 §5.3）。
- host 侧 `kv_.pos` 只作为每轮回读后的镜像，用于容量守卫与断言。

## 3. MTP KV：独立命名空间

- Engine 持有第二个 `KVCache` 实例（`full_layers = 1`，同 dtype/布局/容量）。
  `KVCache` 类型已按层数参数化，无需新类型。
- 位置轴与 target 逻辑位置对齐；提交语义与 §2 完全相同（逻辑回退到 `L+a+1`，
  物理覆盖由下一轮 shifted pass 保证；覆盖先于读取由 fill-then-attend 保证，
  论证见 round-algorithm §2.3）。
- MTP KV 位置 `p` 的条目语义固定为「由 (token `p+1`, hidden at `p`) 产生的 K/V」，
  即 draft 视角整体右错一位；target 与 MTP 的 KV 永不混写。
- `Engine::prefill` 开始时与 target KV 一同 reset。

## 4. Target GDN 状态：snapshot 槽位 + commit-copy

### 4.1 缓冲布局（取代现有单份状态）

每个 GDN 层的状态缓冲扩展为 `S = k+1` 个槽位，**slot 0 即 committed 状态**：

```text
conv_states[l]: [10240, 3, S]   bf16    # slot j = [:, :, j]
ssm_states[l]:  [128, 128, 48, S] fp32  # slot j = [..., j]
```

- 定义：round 之间，slot 0 恒等于与 committed 前缀 `0..L-1` 一致的状态；
  slot `1..k` 内容未定义（上一轮残留）。
- 现有非 verify 路径（chunked prefill、普通 decode 回退路径、reset）一律只
  读写 slot 0，行为与今天完全一致。
- `GdnState::reset` 只需清零 slot 0（其余槽位总是先写后读）。

### 4.2 verify 阶段的写入规则

verify 处理 `T_v = k+1` 个 token 时，conv/recurrent 算子以 slot 0 为初始状态，
在处理完第 `j` 个 token 后把状态快照写入 slot `j`（`j = 0..k`）：

```text
slot[j] = state after [t0, d1..dj] 全部折入   （j=0 即 after-t0）
```

slot 0 被 after-t0 状态覆盖是安全的：`t0` 永远属于 committed 序列（`a ≥ 0`）。
kernel 在写 slot 0 之前已把初始状态读入寄存器（§5.1/§5.2 契约要求），无
读写冒险；这与 vLLM `inplace_final_state=True` 的别名行为相同。

### 4.3 commit-copy

accept 之后、下一次 GDN 读之前，执行一次由 device 标量 `a` 驱动的 gather-copy：

```text
for l in 48 层:  ssm_states[l][..., 0] = ssm_states[l][..., a]
                 conv_states[l][:, :, 0] = conv_states[l][:, :, a]
（a == 0 时为恒等，可在 kernel 内短路）
```

流量上界：`48 × (3.0 MiB + 60 KiB) × 2 ≈ 300 MiB` ≈ 0.17 ms @ 1.79 TB/s，
折合每 token < 0.05 ms，可接受。实现形态（每层一 kernel 或指针表驱动的单
kernel）由实现计划决定，本文只约束语义与时序（round-algorithm §2.2）。

### 4.4 备选方案与否决理由（记录，避免复议）

| 方案 | 否决理由 |
|---|---|
| vLLM 式 indirect-read（下一轮 kernel 按 `a` 选初始槽，无 copy） | 可行且更省带宽，但破坏「round 之间 slot 0 = committed」强不变量，状态 dump/断言/parity 工具全都要带上 `a` 上下文；v1 以可调试性优先，留作性能后续项（契约兼容：只需给 §5 算子加初始槽位标量输入） |
| 快照-回滚-重放（保存 committed，拒绝后重放 accepted 前缀） | 重放需重跑整轮 forward，或按层缓存 conv 输入/q/k/v/g/β 再二次发射 96+ 个 kernel；kernel 序列依赖 `a`，违反总领 C5（不可 graph 化），且拒绝轮延迟尖刺 |
| 单 shadow 状态 + 全量验证后择一 | 只能表达 a∈{0,k} 两种结果，无法部分接受 |
| vLLM 加宽 conv 窗口（`[10240, 3+k]` + offset 读） | 与 per-token 快照语义等价；为使 conv 与 ssm 契约同构、减少一种 kernel 状态寻址模式，统一用快照槽。若实测有利可作为实现细节替换，不改变本文语义 |

### 4.5 内存

| 项 | 每层 | ×48 总量，k=4 | ×48 总量，k=5 |
|---|---|---|---|
| ssm 槽位（含 slot 0） | `3.0 MiB × (k+1)` | 720 MiB | 864 MiB |
| conv 槽位（含 slot 0） | `60 KiB × (k+1)` | 14.1 MiB | 16.9 MiB |
| ssm 相对现状增量（新增 k 槽） | `3.0 MiB × k` | +576 MiB | +720 MiB |
| conv 相对现状增量（新增 k 槽） | `60 KiB × k` | +11.3 MiB | +14.1 MiB |

`k` 必须在 engine load 时固定，进入 cache arena 预算公式。

---

## 5. 算子级状态契约（normative，取代 Part 2 §9.2）

Part 2 §9.2 的 `state_in/state_out` 单终态契约不足以表达部分接受（只有全段 T
之后的终态），予以取代。以下契约中 `S = k+1`，`T ≤ S`；算子不理解 accept，
只做纯序列变换 + 快照。

### 5.1 conv 序列快照

```text
causal_conv1d_sequence_snapshot(
    x:       [10240, T] BF16 compact,
    weight:  [10240, 4] BF16,
    states:  [10240, 3, S] BF16 in-out,     # states[:,:,0] 为初始窗口
    out:     [10240, T] BF16 compact)

语义: 初始窗口 = states[:,:,0]（先读入寄存器）;
      逐 token t: out[:,t] = silu(conv(window, x[:,t])), 窗口右移一列;
      写 after-token-t 窗口到 states[:,:,t]。
```

### 5.2 gated delta recurrent 快照

```text
gated_delta_rule_recurrent_snapshot(
    q: [128,16,T], k: [128,16,T], v: [128,48,T] BF16 compact,
    g: [48,T], beta: [48,T] FP32 compact,
    scale: 1/sqrt(128),
    states: [128,128,48,S] FP32 in-out,     # states[...,0] 为初始状态
    out:    [128,48,T] BF16 compact)

语义: 初始 state = states[...,0]（先读入寄存器/局部）;
      逐 token 执行现有 recurrent 数学（衰减、delta 写、输出）;
      after-token-t 状态写 states[...,t]。
```

现有 `causal_conv1d_prefill/decode`、`gated_delta_rule_recurrent/chunked` 保持
不变（slot 0 就地读写），供 prefill 与回退路径使用。

### 5.3 attention 追加/单查询（device 标量窗口）

```text
gqa_attention_append(q [256,24,T], k [256,4,T], v [256,4,T],
                     base: device I32 标量（窗口基址）,
                     cache: caller-owned KVCache 层切片,
                     out [256,24,T])
语义: 先写 K/V @ base..base+T-1（fill-then-attend）, 行 j attend 0..base+j;
      不推进 KVCache::pos。target verify 与 MTP shifted pass 共用（传各自 cache）。
```

现有 `gqa_attention_prefill(cache_offset)` 是其 host-offset 正确性实现，可先用于
eager 阶段；graph 化需要 device 标量变体。`gqa_attention_decode` 已满足 MTP AR
step 的需求（device `pos`）。small-T 性能形态维持 Part 2 §8 的分类结论。

### 5.4 GDN commit-copy

```text
gdn_commit(conv_states[l], ssm_states[l] for all l,
           a: device I32 标量)
语义: slot[a] → slot[0]（两类状态、48 层）；a==0 时无效果。
```

### 5.5 确定性要求

1. **快照轨迹 ≡ 逐 token 调用**：§5.1/§5.2 算子给定相同输入序列时，
   `states[..., j]` 必须与「以相同 kernel 连续 j+1 次 T=1 调用」的终态**逐 bit
   相等**。可达成依据：recurrent kernel 本就串行遍历 token、状态在寄存器中以
   fp32 演化，store/load fp32 往返无损；conv 窗口只是输入列的搬运。此性质是
   strict-sequential 验证模式（总领 §8.2）的基石，必须有数值测试锁定。
2. 层输入本身（projection 输出）在 batched 与 T=1 路径间允许 kernel 级浮点差异；
   端到端影响按总领 §8.2 的 near-tie 规则验收。
3. argmax tie-break 全项目统一为最低 index。

---

## 6. Engine 侧新增缓冲与标量汇总

| 缓冲/标量 | 形状/类型 | 生命周期 |
|---|---|---|
| verify hidden | `[5120, k+1]` bf16 | 每轮覆盖；shifted pass 的 hidden 输入 |
| prefill chunk hidden | `[5120, prefill_chunk]` bf16 | 每 chunk 覆盖 |
| verify logits | `[248320, k+1]` bf16 | 每轮覆盖（取代 `StepState.logits [vocab,1]` 的单列限制） |
| `target_tokens` | `[k+1]` i32 device | verify → accept |
| `drafts` | `[k]` i32 device | propose → 下一轮 verify |
| `sampled_out` / `num_sampled` | `[k+1]` i32 / i32 device | accept → host 回读 |
| `verify_ids` / `shifted_ids` | `[k+1]` i32 device | 轮内 |
| positions 缓冲 | `[k+1]` i32 device | verify 填写，shifted pass 复用 |
| `L`（committed length） | i32 device 标量 + host 镜像 | 跨轮 |
| `window_base`（轮初 L 快照） | i32 device 标量 | 轮内（verify/shifted pass 窗口基址） |
| `a` / `ar_pos` | i32 device 标量 | 轮内 |
| MTP AR hidden | `[5120,1]` bf16 | AR step 间 |
| 统计计数器 | 若干 i64 device | 跨轮累计 |

这些缓冲全部进入 cache arena / `StepState` 扩展；`default_cache_bytes` 与
`default_work_bytes` 公式必须同步（workspace 峰值新增 MTP shifted pass 的
`[34816,k+1]` gateup 等项，见 implementation-requirements §5）。

---

## 7. 状态调试与 parity 工具约定

1. GDN slot 0 与 MTP KV 纳入现有 dump/parity 体系（FileTap / layer_dump 级别），
   round 之间 slot 0 可与 ref model 的状态直接对拍（ref model 用 KV truncate +
   顺序 verify，与本约定在 committed 语义上等价）。
2. strict-sequential 模式下（round-algorithm §3.2）不使用快照槽，状态即
   committed，可与 MTP-off 基线逐 bit 对拍——用于隔离「算法/状态机 bug」与
   「batched kernel 数值差异」。
3. ref model `forward_mtp_verified` 保持为算法 oracle：它验证的性质（MTP on/off
   greedy 输出一致、接受率统计）就是 strict-sequential 模式的验收标准。
