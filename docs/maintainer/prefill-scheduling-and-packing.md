# Prefill 调度、打包与默认 chunk

本文记录 `ninfer-35b-prefill-2x` 任务引入的长期合同。验收口径见任务 Spec；
性能数据和对比记录在 `.agent-tmp/ninfer-35b-prefill-2x/`，不在此重复。

## 默认 prefill_chunk（Qwen3.5）

- 唯一权威来源：`src/models/qwen3_5/prefill_chunk.h`。
  `kResidentPrefillChunk = 2048`，`kHostExpertsPrefillChunk = 4096`。
- `resolve_prefill_chunk(options, placement)`：显式值优先，结果取 `min(值, max_context)`，
  非 128 的倍数抛 `std::invalid_argument`。
- `Startup::plan` 解析后回写 `EngineOptions::prefill_chunk`，所以
  `engine.options().prefill_chunk == engine.description().prefill_chunk == ExecutionPlan::chunk`。
- Sequence planner 只接受已解析的值，`prefill_chunk == 0` 时抛
  `prefill_chunk must be resolved before sequence planning`。
- 该默认值对同一架构入口的其他 checkpoint（如 27B）同样生效。

## Scheduler 的 prefill 节拍

`Scheduler::configure_prefill(PrefillPacing)` 取代了旧的 `PrefillPolicy`。
唯一定义节拍常量的位置是 `EngineCore::prefill_pacing(pack_tokens, chunk, concurrency)`。
`pack_tokens` 来自 `Session::prefill_pack_tokens()`。

每个周期由 `begin_prefill_cycle(decode_waiting)` 选择工况。
`decode_waiting` 表示存在未被 capture 阻塞的 decode-ready 或 control-ready lane。

| 工况 | 条件 | 预算 |
| --- | --- | --- |
| Throughput | 没有等待 decode 的 lane | `max(pack_tokens, chunk)` token，单元数 `max(2, concurrency)`，允许多段打包 |
| Interleave | 有等待 decode 的 lane | 按预测耗时放行：`interleave_ns`（100 ms）与 `2 × chunk` token 两个上界，另有 debt cap 300 ms、repay 4 ms |

- 每周期首个单元总是放行。放不下的 lane 结束本周期，游标停在该 lane。
- 耗时预测：`prefill_executed(lane, concurrency, tokens, elapsed_ns)` 维护按 lane 的
  ns/token EMA，打包 flight 的耗时按 token 比例分摊。没有估计的冷启动 lane 使用
  `interleave_cold_tokens`（`min(chunk, 1024)`）。lane 释放或换绑时调用 `prefill_lane_released`。
- `prefill_unit_caps` 的结果在 `reserve_resident_units` 中写入 Prefill/Replay 的
  `ExecutionUnit.tokens`，并固定在本周期 permit 中。该字段是 prompt 位置上界，0 表示 Session 窗口。
  Session 取 `min(window_end, cursor + tokens)`，已有 permit 时 frontier 取 min。
- 单元粒度 `unit_granularity = 128`。
- 准入规则（FIFO、bypass 额度、单 binding 在途）、progress owner 优先、Vision 连续性和打包互斥未改变。
- 每个周期的 prefill 工作量上界由 `prefill_tokens_remaining` 与 `prefill_ns_remaining` 决定。
  k 个 prefilling lane 在有限周期内都被推进（`tests/runtime/test_scheduler.cpp` 覆盖 k=2..6）。

## Prefix checkpoint 捕获与回收不阻塞

- 三个模型的 Session 新增 `terminal_blocked(SequenceHandle)`，是 `finish/abort` 能否执行的唯一判断。
  Qwen3.5 只在本 lane 被上下文操作占用或执行冲突时返回 true；qwen4_exp、glm5next 保守返回
  `has_context_operation()`。
- Engine 的结算和取消路径逐 lane 检查 `terminal_blocked`，不再因存在上下文操作而全局早退。
  上下文操作在途时，ResourceManagement 的 publish/finish/abandon/release_source 不被调用，
  结算记入 `deferred_settlements_`，在 `begin_cycle` 中按 `publication_order` 先进先出补做。
- Session 新增 `context_epoch()` 和 `context_pins(handle)`。Qwen3.5 在目录、池占用、placement 清单变化时
  递增 epoch；qwen4_exp、glm5next 的 `context_pins` 在 `has_context_operation()` 时恒为 true。
- `ResourceManager::begin_reclaim` 与评估结果按 `(context_epoch, 自身 mutation epoch)` 缓存复用；
  `planning_stats()` 报告构建、复用、选择次数和规划耗时。
- 上下文操作期间 `reclaim` 只在未被 pin 的缓存 checkpoint 中做纯释放，仍遵守 `may_revoke` 与
  `ReclaimRights`。降级、维护、暂停不在上下文操作期间启动。
- Qwen3.5 `reserve_units(Decode)` 先预留 `frontier + kDecodeGrowthReserveTokens`（1024，Session 内部常量），
  不可行时退回最小需求；最小需求也不可行时，assessment 以视界需求为目标，一次回收覆盖多步 decode。
- prefix reuse 和 `checkpoint_interval` 的默认值未改动。

## 多段 prefill 打包

- `Session::prefill_pack_limit()`（段数上限，来源是 Program 规划的 `pack_segments`）与
  `Session::start_prefill(std::span<const SequenceHandle>, std::span<const bool>)` 返回
  `PrefillStart{ticket, segments}`，接纳非空前缀。旧的单段 `start_prefill` 已删除。
- 接纳规则归 Session：多段时所有段都必须是 `SamplePurpose::None`，非 Vision 窗口，后端非 DFlash，
  且 Σlen ≤ `resolve_prefill_pack_tokens`（`prefill_chunk × 2`，上限为对齐后的 max_context）。
  FirstToken、Vision、DFlash 段单独成包。qwen4_exp、glm5next 的上限为 1。Replay 仍走 `start_replay`。
- 在 cap 截断下段长可以不等（含 1、非 128 倍数），范围是 1 到 chunk。
- 每个 flight 只有一次 Commit。`pending` 只可能出现在 `segments == 1` 的 flight 中。
- 前向：embedding、norm、linear、MoE 在 Σtokens 上合并计算。GQA 与 GDN（含 conv 状态）通过
  `PackedSequenceUpdate` 对每段调用既有 B=1 Op，使用各段自己的 table row、positions 和 state offsets。
  没有新增 Op 合同。
- 设备内存（token 维 workspace、frame、段元数据）在规划阶段按 pack_tokens 申请，
  mixer scratch 与 mtp_prefill 仍按 chunk 规划，执行期不调用 `cudaMalloc`。
- 存在 progress owner 优先的 prefill 或 Vision 连续性 lane 时，flight 只含该 lane。

## Op 内核

公开 Op 签名和支持范围不变；workspace 字节数通过各 Op 的 capacity 查询在规划阶段生效。
decode 小 W（≤16）和 MTP verify 路径不受影响。

- FP8 KV 因果 attention 的 Tiled MMA 路径（`mxfp8_tiled_mma.cuh`、`schedule.cuh`、`common/cuda/mma.cuh`）。
- groupwise_int MoE 的 `grouped_gate_up`、`grouped_down`、`grouped_swiglu`、`grouping`。

## 验证入口

| 内容 | 位置 |
| --- | --- |
| Scheduler 节拍与公平性 | `tests/runtime/test_scheduler.cpp`、`tests/runtime/test_t5_flight_scheduling.cpp` |
| 回收缓存与 pin | `tests/runtime/test_resource_manager.cpp`、`tests/models/qwen3_5/test_native_transactions.cpp` |
| 默认 chunk 单一来源 | `tests/models/qwen3_5/test_t1_prefill_chunk_default.cpp`、`tests/test_t1_prefill_chunk_default.py` |
| 捕获不阻塞与取消回收 | `tests/models/qwen3_5/test_t3_capture_nonblocking.cpp` |
| 打包合同与模型级一致性 | `tests/models/qwen3_5/test_t6_packed_contract.cpp`、`test_packed_prefill_real.cpp` |
| 大 W 与 decode 小 W attention | `tests/test_t4_large_w_prefill_f6.py`、`tests/test_t4_decode_width_ops.py` |
| 贪心输出与 PPL | `tests/test_t4_serve_greedy_ppl.py` |
| 打包基准 | `bench/models/qwen3_5/packed_prefill_bench.cpp`（记录 `segments, segment_tokens, total_tokens, packed_ms, sequential_ms`） |

## 已过时的稳定文档

下列文档按 AGENTS.md 不随实现自动更新，需要用户安排修订：

- `docs/startup-configuration.md:86`：写着 "other paths to 1024"，Qwen3.5 Resident 路径现在是 2048。
- `docs/maintainer/qwen3_5-model.md`、`engine-architecture.md`、`context-management.md`：
  尚未描述 prefill 节拍、`terminal_blocked`、延后结算、回收缓存和多段打包。
