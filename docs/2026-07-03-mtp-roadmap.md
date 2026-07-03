# MTP Landing Roadmap

> Status: normative sequencing（里程碑顺序与边界的权威来源）。Date: 2026-07-03。
> 这是 roadmap，不是实现计划：它规定做什么、按什么顺序、边界在哪、以什么判据算完成。
> 每个里程碑启动时在 `docs/plans/` 下另立一个正式 plan（按 AGENTS.md 计划规范，
> subagent-driven），任务锚定
> [implementation-requirements](2026-07-03-mtp-implementation-requirements.md) 的需求 ID。
> 算法与状态语义以[总领文档](2026-07-03-mtp-spec-decode-overview.md)及其细分文档为准。

## 端态（roadmap 完成时的状态）

- 引擎支持 `--mtp-draft-tokens k`（k ∈ [1,5]，load 时固定）；默认值由 M5 的 bench
  数据选定。关闭时行为与现状完全一致。
- greedy 语义下：strict-sequential 模式输出与 MTP-off 逐 token 相等；batched 生产
  路径通过 near-tie 验收（总领 §8.2），并保留 strict-sequential 作为常驻调试开关。
- 整个 verify+propose round 可作为固定 kernel DAG 捕获进 CUDA graph；
  `qus_bench` 输出 MTP 段（acceptance rate/length、rounds、tok/s），有 bench 报告
  证明相对 MTP-off 的端到端加速。
- 新 v3+MTP 工件是唯一权重工件；不存在"无 MTP 版工件"兼容路径。

## 里程碑总览

```text
M0 工件与工具链解锁
 ├──► Track A: M1 W8G32 线性算子族 ──► M2 MTP head C++ 前向
 └──► Track B: M3 target verify 路径与状态底座
                （A、B 并行，文件所有权互不相交，见"协调点"）
M2 + M3 ──► M4 引擎 round 循环与正确性验收 ──► M5 性能化
```

| 里程碑 | 范围（需求 ID） | 产出 | gate（完成判据） |
|---|---|---|---|
| **M0 工件与工具链解锁** | R-FMT-1/2/3 | C++ 可加载 v3+MTP 工件；W8G32 打包/夹具/bench 解析可用 | 新工件在 `load_mtp` 开/关下均可加载；现有全部测试与 greedy parity 在新工件上不变绿 |
| **M1 W8G32 线性算子族** | R-L1-1 | W8G32 codec + T=1/small-T/LargeT linear，5 个 shape family | 按 l1-op-test-standard 数值通过（T=1,2,5,17 + 真实 shape）；bench 基线入档 |
| **M2 MTP head C++ 前向** | R-L2-1/4、R-L0-2、R-L1-2、R-L2-5(MTP 项) | MTP 绑定 + `mtp_forward`（shifted pass / AR step）+ MTP KV | C++ 与 ref model `mtp_forward` 逐层 parity（真实权重，prefill 与 decode 两相位）；greedy draft token 一致 |
| **M3 target verify 路径与状态底座** | R-L0-1/3/4、R-L1-4/7、R-L2-2/3、R-L2-5(verify 项) | KV rewind、GDN 快照槽与快照算子、Verify 相位（T=k+1 全列 hidden/logits/argmax）、prefill 全列 hidden | 快照轨迹与逐 token 调用逐 bit 相等；verify 窗口 vs 顺序复放 parity（T=1..6，op 容差）；sanitizer 干净；记录 verify 成本 checkpoint |
| **M4 引擎 round 循环与验收** | R-RT-1..5/7、R-L1-5/6、R-TOOLS-1、R-TOOLS-2(schema) | 完整 round 状态机、prefill 期 MTP KV 填充、strict-sequential 模式、统计与 bench schema | strict-sequential == MTP-off 逐 token 相等（canonical prompts，k=1..5）；batched 通过 near-tie 规则；acceptance 统计与 ref model 一致；k=5 内存预算成立；sanitizer E2E 干净 |
| **M5 性能化** | R-RT-6、R-L1-3(device 标量/small-T 调优)、R-TOOLS-2(报告)、Part 2 Class C/Phase 2-3 候选 | round 级 CUDA graph、small-T 调优 kernel、k 默认值、bench/nsys/ncu 证据报告 | 端到端 tok/s 相对 MTP-off 的实测加速入档；总领 §7.2 模型与实测对账；graph 路径与 eager 输出一致 |

## 各里程碑边界

### M0 — 工件与工具链解锁

- **做**：解析器 tag 6 + W8G32 组/平面尺寸表（parser 与 weight_store 两处）；
  `Engine::load` 启用 `load_mtp` 与 MTP expectations、weight arena 预算；
  W8G32 测试打包器、fixture 生成、`linear_op_bench` qtype 解析。
- **不做**：任何 W8 kernel、任何 MTP 绑定或执行代码。
- 说明：这是唯一的硬阻塞项——当前 C++ 连纯文本推理都无法在新工件上运行。规模小、
  收益立竿见影，必须最先落地。

### M1 — W8G32 线性算子族（Track A）

- **做**：K-group 32 codec；generic GEMV（T=1）、small-T multistep、LargeT 路径；
  注册 `[5120,10240]`、`[14336,5120]`、`[5120,6144]`、`[34816,5120]`、
  `[5120,17408]` 五个 family；数值测试与 bench 扫描。
- **不做**：fused epilogue、直写 compact 输出、深度调优（全部 M5）；不碰 model card。
- 正确性优先：本里程碑的性能产出只是基线数据，不设吞吐 gate。

### M2 — MTP head C++ 前向（Track A，依赖 M0+M1）

- **做**：`ModelConfig`/绑定结构（`MtpW`）；第二个 `KVCache` 实例；
  `mtp_pack_fc_input` 与 `attn_in` compact split；`mtp_forward` 两个入口
  （shifted pass 用现有 `gqa_attention_prefill(cache_offset)` 作 append，AR step 用
  现有 decode attention）；workspace/cache 公式的 MTP 项。
- **不做**：engine round 逻辑、target 侧任何改动、draft/verify 状态机。
- 可独立验证：给定相同 (ids, hidden, positions)，与 ref model `mtp_forward` 对拍，
  不需要 round 循环存在。

### M3 — target verify 路径与状态底座（Track B，仅依赖 M0）

- **做**：`KVCache` 回退 API；`GdnState` 扩展为 k+1 槽位（slot 0 = committed）；
  快照算子对（`causal_conv1d_sequence_snapshot`、
  `gated_delta_rule_recurrent_snapshot`）与 `gdn_commit`；T=1-only fused 算子的
  T>1 generic 回退；`run_layers` 新增 Verify 相位（T=k+1，末端全列 final norm +
  lm_head + 逐列 argmax）；prefill chunk 全列 hidden 输出；`StepState`/缓冲/标量
  底座；workspace/cache 公式的 verify 项。
- **不做**：accept/commit 判定逻辑（属 M4）、MTP、任何 W8（verify 走 TEXT_CORE
  Q4/Q5/Q6，与 Track A 无依赖）。
- **附带 checkpoint**：用 generic 路径实测 verify 窗口（T=2..6）相对 T=1 decode 的
  成本比 ε，回填总领 §7.2 的加速模型——这个数字决定 M5 调优的优先级排序，也提前
  暴露"speculative 不划算"的风险。

### M4 — 引擎 round 循环与正确性验收（依赖 M2+M3）

- **做**：round 状态机（verify → accept/commit → propose → host 回读）与容量守卫；
  accept kernel、ids/positions 装配 kernel、行 a gather；prefill 期逐 chunk 的
  MTP shifted pass（含 `next_prefill_token`）；`EngineOptions.mtp_draft_tokens` 与
  CLI；`generate`/text runner 多 token 交付与 stop/overshoot 语义；位置记账统一
  （消除 decode_step/record 的 advance 不对称）；strict-sequential 调试模式；
  统计计数器 + bench schema 的 MTP 段；状态 dump 对拍入口；canonical fixture E2E
  测试与 sanitizer E2E。
- **不做**：round 级 CUDA graph、任何 kernel 调优。eager round 即可过 gate。
- 这是唯一的集成里程碑：算法正确性的全部验收都压在这里，gate 从严。

### M5 — 性能化（依赖 M4）

- **做**：attention append 的 device 标量窗口基址变体与整轮 CUDA graph 捕获；
  按 M3/M4 的 profiling 证据选做 small-T 调优（GQA append small-T、Q4/Q5 small-T
  fused、W8 tuned/fused epilogue，即 Part 2 Class C 与 Phase 2/3 候选）；可选的
  indirect-read commit 替换 commit-copy；k 默认值由 bench corpus 数据选定；
  nsys/ncu + `qus_bench` 证据报告。
- **不做**：算法变更（概率采样仍在范围外）；未经 profiling 证据不开工任何
  strict one-launch 实验（Part 2 §14.4 的原则继续适用）。
- gate 是实测数字入档，不是预设吞吐值；若实测加速不达预期，以 M3 checkpoint 与
  profiling 数据决定继续调优或调整 k/策略。

## 顺序与边界的理由

- **M0 最先**：它解除的是"任何人都无法在新工件上跑任何东西"的全局阻塞，且与后续
  一切工作正交。
- **两条并行 track**：Track A（W8 → MTP head）与 Track B（target 状态与 verify）
  的文件所有权几乎不相交——A 在 `linear/`、MTP 绑定与新算子；B 在
  `kv_cache/state_store`、GDN kernel 与 verify 调度。并行不引入集成风险，汇合点
  只有 M4。
- **verify（M3）先于 round（M4）**：GDN 快照语义是全项目风险最高的新机制，必须在
  没有 round 状态机干扰的情况下用"bit 相等 + 顺序复放 parity"单独钉死；M4 才能把
  失败归因收敛到胶水逻辑。
- **正确性（M4）先于性能（M5）**：与 AGENTS.md 的验证观一致——strict-sequential
  基线先立住，batched 与 graph 路径才有可对拍的锚；调优必须以 profiling 证据立项。
- **不再细分**：每个里程碑是一个可独立验证的行为边界（工件可加载 / 算子族数值
  正确 / MTP 前向对拍 / verify 窗口对拍 / E2E 输出相等 / 实测加速）。里程碑内部的
  任务拆分留给各自的正式 plan，不在 roadmap 层展开。

## 协调点（并行期间的共享文件）

| 共享面 | 涉及 | 约定 |
|---|---|---|
| `src/model/qwen3_6_27b.cpp` / `model.h` | M2（新增 mtp_forward/绑定）与 M3（run_layers Verify 相位） | 区域不相交：M2 只做加法（新函数/新结构），M3 改 `run_layers`/prefill；先落地者不重构对方区域 |
| workspace/cache 预算公式（`engine.cpp`） | M2（MTP 项）与 M3（verify/快照项） | 各自追加独立的相位项，汇总在 M4 复核一次 |
| `StepState` 与 device 标量 | M3 建底座，M2/M4 消费 | 布局以 state-management §6 为准，M3 一次定型 |
| CMake/测试注册、`q5090_pack.h` | M0/M1/M3 | 机械冲突，按落地顺序合并 |

## 风险与检查点

1. **small-T verify 成本（最大性能风险）**：若 T=6 verify 成本远超 T=1（generic
   路径 launch 开销累积），加速模型坍塌。M3 的成本 checkpoint 提前量化，M5 据此
   排序调优项。
2. **near-tie 分叉验收**：batched 与 strict-sequential 的分叉规则（总领 §8.2）在
   M4 首次接受实战检验；若分叉频率异常，先查算子 parity 再放宽阈值，不得静默调参。
3. **内存预算**：k=5 时快照槽 +720 MiB（state-management §4.5），M4 在
   max_ctx=8192 下验证整机预算；若紧张，k 上限收缩优先于机制变更。
4. **acceptance 数据代表性**：现有接受率证据来自单 prompt；M4/M5 用 bench corpus
   重测后才能定 k 默认值与宣称加速比。
