# MTP 实现需求清单（实现需求文档）

> Status: normative requirements，隶属于 [总领文档](2026-07-03-mtp-spec-decode-overview.md)。
> Date: 2026-07-03。基线 commit `bd1d149`。
> 本文列出把总领 C1–C5 约定落进引擎所需的全部改动，按层分组、带依赖顺序与
> 验证要求。它不是实现计划：正式计划后续在 `docs/plans/` 下按 AGENTS.md 计划
> 规范（subagent-driven）另立，任务拆分应以本文需求 ID 为锚。

现状结论（详见调研）：MTP 在 C++ 树中为零实现；且**当前 C++ 解析器无法加载新
的 v3+MTP 工件**（W8G32 qtype tag 6 被拒），即使纯文本推理也会失败——R-FMT-1
是最高优先级阻塞项。

---

## 1. R-FMT — 格式与加载（阻塞项）

| ID | 需求 | 现状锚点 |
|---|---|---|
| R-FMT-1 | C++ 解析器接受 `W8G32_F16S`（tag 6）：`qtype_from_tag`、quant 判定、group/plane 尺寸表（parser 与 weight_store 两处） | `src/core/weight_store_parser.cpp:118-128,243-286`；`src/core/weight_store.cpp:185-210` |
| R-FMT-2 | `Engine::load` 启用 `load_mtp`（由 EngineOptions 的 MTP 开关驱动）；MTP module 的 expectations 校验（12 blocks/16 segments/2 fusion groups、W8G32 five linears、BF16 norms）；weight arena 预算含 MTP payload（≈431 MiB） | `include/qus/core/weight_store.h:18`；`src/runtime/engine.cpp:134-141,286-288` |
| R-FMT-3 | 测试打包器、fixture 生成器、bench qtype 解析支持 W8G32；fixture 覆盖 MTP blocks | `tests/kernels/q5090_pack.h`；`tests/fixtures/make_q5090_fixture.py`；`bench/linear_op_bench.cu` |

注：W8 组参数以 v3 格式文档与已验证工件为准（**G32**）；Part 2 §5 按 G128 写的
payload/codec 参数作废，需按 `kGroupK=32`、`code 32B + scale 2B / group` 重推。

## 2. R-L0 — 核心基础设施

| ID | 需求 | 对应约定 |
|---|---|---|
| R-L0-1 | `KVCache` 回退：将 `pos` 置为 ≤ 当前值的目标值；现有 `position >= pos` 读守卫即回退语义 | C2 |
| R-L0-2 | Engine 持有第二个 `KVCache`（1 层，MTP 命名空间），随 prefill reset | C4 |
| R-L0-3 | `GdnState` 扩展为每层一个 conv 大 tensor 和一个 ssm 大 tensor，逻辑 `k+1` 槽位（conv `[10240,3,Slots]`、ssm `[128,128,48,Slots]`）；`reset` 清 slot 0；selected-state 路径由 `gdn_initial_slot` 指定初始槽位 | C3；state-management §4.1 |
| R-L0-4 | `StepState`/device 标量扩展：`logits [248320,k+1]`、`target_tokens/drafts/sampled_out/verify_ids/shifted_ids/positions` 缓冲、`L/a/ar_pos/num_sampled/gdn_initial_slot` 标量、统计计数器、verify hidden `[5120,k+1]`、chunk hidden `[5120,prefill_chunk]`、AR hidden | C5；state-management §6 |

## 3. R-L1 — 算子

Part 2 的分类（Class A/B/C）与优先级矩阵继续有效，以下为约定层面的增补与修正：

| ID | 需求 | 说明 |
|---|---|---|
| R-L1-1 | W8G32 row-split linear 族：codec（K-group 32）+ T=1 GEMV + small-T（T≤16）+ LargeT；注册 shape families `[5120,10240]`、`[14336,5120]`、`[5120,6144]`、`[34816,5120]`、`[5120,17408]` | Part 2 §5（Class A），组参数改 G32 |
| R-L1-2 | `mtp_pack_fc_input`（`[5120,T]×2 → [10240,T]`）与 `attn_in [14336,T]` 的 compact split（Option A 起步） | Part 2 §6/§7（Class B） |
| R-L1-3 | GQA append 原语：使用统一 `gqa_attention(..., positions, ...)`，由 device positions 决定写入/窗口；small-T 调优按 Part 2 §8.2 | C5；state-management §5.3 |
| R-L1-4 | GDN 快照算子对：`causal_conv1d_sequence_snapshot`、`gated_delta_rule_recurrent_snapshot` 均接收 `initial_slot` device 标量，读取 selected 初始槽并顺序写 slot `0..T-1`（契约见 state-management §5.1/§5.2，**取代 Part 2 §9.2 单终态契约**）；MTP round helper 提供 `gdn_initial_slot` reset/set 小 kernel | C3 |
| R-L1-5 | accept kernel：输入 `target_tokens/drafts`，输出 `a/t*/sampled_out/num_sampled`，更新 `io_.token/L`、统计计数器（单 block 小 kernel） | C1/C5 |
| R-L1-6 | verify_ids / shifted_ids / positions 装配 kernel（round-algorithm §2.1/§2.3 的索引规则；含 device 行号 gather `mtp_hidden[:,a]`） | C4/C5 |
| R-L1-7 | T=1-only fused decode 算子（`gdn_in_vz_decode` 等）在 verify 路径的 T>1 替代：先 generic `linear()`+elementwise 正确性回退，性能项按 Part 2 Class C | Part 2 §4.3 |

## 4. R-L2 — model card 与调度

| ID | 需求 | 说明 |
|---|---|---|
| R-L2-1 | `ModelConfig` 增补 MTP 常量；`MtpW` 绑定结构（fc、attn_in.w8、o_proj、gateup.w8、down、7 个 norm），经 `(ModuleKind::MtpDraft, source_kind/fused)` 解析——WeightStore 查找机制已就绪 | Part 1 布局 |
| R-L2-2 | verify 调度：`run_layers` 新增 Verify 相位（T=k+1；full-attn 用 append 原语 @ device base；GDN 用快照算子；末端全列 final norm + lm_head + 逐列 argmax） | C1/C2/C3 |
| R-L2-3 | prefill chunk 全列 final-norm hidden 输出（写 chunk hidden 缓冲；logits 仍只算最后列） | round-algorithm §1.1 |
| R-L2-4 | `mtp_forward`：shifted pass（T=k+1，fill-then-attend 写 MTP KV，仅需单列 logits）与 AR step（T=1，decode 式 attention）两个入口 | C4；round-algorithm §1.2/§2.3/§2.4 |
| R-L2-5 | `default_work_bytes`/`default_cache_bytes` 公式同步：MTP 相位的 workspace 峰值（gateup `[34816,k+1]`、fc 输入 `[10240,k+1]` 等）与 cache arena 新增（快照槽、MTP KV、k+1 列 logits/hidden） | state-management §4.5/§6 |

## 5. R-RT — 引擎与前端

| ID | 需求 | 说明 |
|---|---|---|
| R-RT-1 | round 循环：verify → accept/selector-update → propose → host 回读；容量守卫 `L + 2k ≤ max_ctx`，否则回退 T=1 decode；MTP-enabled fallback 从 selector 读 GDN 状态、写 slot 0、再 reset selector；round 内禁止 `prefill()` | round-algorithm §2/§3 |
| R-RT-2 | `EngineOptions` 增加 `mtp_draft_tokens k`（0 = 关闭；1..5），load 时定死缓冲；CLI 透传（`--mtp-draft-tokens`） | 总领 §7 |
| R-RT-3 | 位置记账：device `L` 为权威，host 镜像每轮回读后同步（`kv_.pos`/`mtp_kv.pos`/`L`）；消除现有 `decode_step` 与 `decode_step_record` 的 advance 不对称问题 | C2/C5 |
| R-RT-4 | `generate`/`TextGenerationRunner` 支持每步多 token 交付：stop token 截断、`max_new` overshoot-and-truncate、流式 detokenize 按 sampled 批推进 | round-algorithm §2.5 |
| R-RT-6 | CUDA graph：v1 允许 eager；但所有数据流必须自始满足 C5（形状只依赖 k、`a` 仅 device 标量），整轮单图捕获为性能后续项 | C5 |
| R-RT-7 | `memory_stats` 与统计出口扩展（rounds、acceptance 指标） | round-algorithm §4 |

## 6. R-TOOLS — 工具链

| ID | 需求 |
|---|---|
| R-TOOLS-2 | `qus_bench` MTP 模式：报告 schema 新增 `mtp` 段（`k`、`rounds`、`fallback_steps`、`draft_tokens`、`accepted_tokens`、`acceptance_rate`、`acceptance_length`、`accepted_per_pos[]`），schema 版本号提升；`linear_op_bench` 增加 W8 shape 扫描 |

## 7. 验证要求（对照 AGENTS.md 测试政策）

按白名单归类，禁止为凑覆盖另立测试：

1. **数值正确性（白名单 1）**：W8G32 linear（真实 K 对齐 + 小合成形状，T=1/2/5/17）；
   GQA append（T=1..6 × cache_offset 若干，含 batch 内因果性与 cache 写位等价）；
   GDN 快照算子的 selected-slot 读取、顺序写槽、slot 0 alias 正确性。
2. **格式契约（白名单 2）**：W8G32 解析（合法/畸形 payload 拒收）、MTP module
   expectations、fixture 端到端 load-and-bind。
3. **CLI/报告契约（白名单 3）**：bench schema 新字段（`qus_bench_support` 契约
   测试延伸）。
4. **端到端行为（白名单 4，canonical fixtures）**：MTP batched 模式在固定 prompt
   下完成请求 token 数、记录 round/acceptance/fallback 统计，并正确处理容量
   fallback、stop token 截断与 `max_new` overshoot。
5. **GPU 内存/生命周期（白名单 5）**：新状态缓冲（快照槽、MTP KV、round 缓冲）
   跑 `compute-sanitizer`（重点：slot 别名写、junk 行 KV 写、rewind 后重写）。
6. 性能验收沿用 l1-op-test-standard §2（ncu DRAM 吞吐阈值）+ `qus_bench`
   acceptance/吞吐报告；不得以打印 GB/s 充当证据。

## 8. 依赖顺序与里程碑

顺序、边界、并行 track、gate 与协调点统一由
[MTP Landing Roadmap](2026-07-03-mtp-roadmap.md) 规定（M0 工件解锁 → Track A:
M1 W8 线性族 → M2 MTP 前向 ∥ Track B: M3 verify 与状态底座 → M4 round 循环与
验收 → M5 性能化）。roadmap 的每个里程碑对应一个 `docs/plans/` 正式计划，任务
以本文需求 ID 为锚；计划内的任务边界按「可独立验证的行为」切，不按文件拆分。

## 9. 需求 ↔ 约定回溯表

| 约定 | 关键需求 |
|---|---|
| C1 验证窗口与接受 | R-L1-5, R-L2-2 |
| C2 KV 逻辑回退 | R-L0-1, R-RT-3 |
| C3 GDN 快照+selector | R-L0-3, R-L1-4 |
| C4 MTP KV + 固定形状 propose | R-L0-2, R-L1-6, R-L2-4 |
| C5 device 标量驱动/graph 友好 | R-L0-4, R-L1-3(标量变体), R-RT-6 |
| 验收契约（总领 §8） | 验证 1/4/5 |
