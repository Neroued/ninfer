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

## Scheduler 的 prefill 周期预算

`Scheduler::configure_prefill(PrefillPolicy{max_units_per_cycle, max_tokens_per_cycle})`。
Engine 将预算设为 `2 × prefill_chunk` token，单元数为 `max(2, concurrency)`。

- 每周期首个单元总是放行。之后的单元要求声明 token ≤ 剩余预算，且单元数未超限。
- 遇到第一个放不下的 lane 就结束本周期，游标停在该 lane，下周期从它开始。
- 计费按 Session 实际回报的 token 数（`prefill_executed(lane, concurrency, tokens)`）。
- 上界：每周期 prefill token ≤ `max(max_tokens_per_cycle, 最大单元)`。有 decode-ready lane 时，
  相邻两次 decode 之间的 prefill 量不超过这个值。
- k 个 prefilling lane 在不超过 k 个周期内都至少推进一个单元。
- 准入规则（FIFO、bypass 额度、单 binding 在途）没有改动。

## Prefix checkpoint 捕获不阻塞

- 三个模型的 Session 新增 `terminal_blocked(SequenceHandle)`，是 `finish/abort` 能否执行的唯一判断。
  Qwen3.5 只在本 lane 被上下文操作占用或执行冲突时返回 true；qwen4_exp、glm5next 保持保守语义
  （有上下文操作即返回 true）。
- Engine 的结算和取消路径逐 lane 检查 `terminal_blocked`，不再因存在上下文操作而全局早退。
- 上下文操作在途时，ResourceManagement 的 publish/finish/abandon/release_source 不被调用，
  相应结算记入 `deferred_settlements_`，在 `begin_cycle` 中无上下文操作、准入之前按
  `publication_order` 先进先出补做。
- prefix reuse 和 `checkpoint_interval` 的默认值未改动。

## 多段 prefill 打包

- `Session::prefill_pack_limit()` 与 `Session::start_prefill(std::span<const SequenceHandle>,
  std::span<const bool>)` 返回 `PrefillStart{ticket, segments}`，接纳非空前缀。
  旧的单段 `start_prefill` 已删除。
- 接纳规则归 Session：多段时所有段都必须是 `SamplePurpose::None`，非 Vision 窗口，后端非 DFlash，
  且 Σlen ≤ `resolve_prefill_pack_tokens`（`prefill_chunk × 2`，上限为对齐后的 max_context）。
  FirstToken、Vision、DFlash 段单独成包。qwen4_exp、glm5next 的上限为 1。Replay 仍走 `start_replay`。
- 每个 flight 只有一次 Commit。`pending` 只可能出现在 `segments == 1` 的 flight 中。
- 前向：embedding、norm、linear、MoE 在 Σtokens 上合并计算。GQA 与 GDN（含 conv 状态）通过
  `PackedSequenceUpdate` 对每段调用既有 B=1 Op，使用各段自己的 table row、positions 和 state offsets。
  没有新增 Op 合同。
- 设备内存（token 维 workspace、frame、段元数据）在规划阶段按 pack_tokens 申请，
  mixer scratch 与 mtp_prefill 仍按 chunk 规划，执行期不调用 `cudaMalloc`。
- 存在 progress owner 优先的 prefill 或 Vision 连续性 lane 时，flight 只含该 lane。

## 大 W attention 内核

FP8 KV 因果 attention 的 Tiled MMA 路径做了优化
（`src/ops/softmax_attention/.../mxfp8_tiled_mma.cuh`、`schedule.cuh`、`common/cuda/mma.cuh`）。
公开签名和 decode 小 W（≤16）路径不变。MoE groupwise_int 内核本次未改动。

## 验证入口

| 内容 | 位置 |
| --- | --- |
| Scheduler 预算与公平性 | `tests/runtime/test_scheduler.cpp`、`tests/runtime/test_t5_flight_scheduling.cpp` |
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
  尚未描述周期预算、`terminal_blocked`、延后结算和多段打包。
