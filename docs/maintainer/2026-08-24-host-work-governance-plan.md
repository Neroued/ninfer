# Engine Host 工作治理与可观测性实施计划

状态：待实施；资源事务子项 M1.3 已完成

资源前置条件已经落入
[资源调度与上下文缓存架构](resource-scheduling-and-context-cache.md)：Program 是唯一物理 authority，
ResourceManager 只拥有逻辑策略，周期性 replica scan 已删除。本文后续工作直接建立在该当前合同上。

适用基线：2026-08-24 当前工作树

适用产品：单 GPU、单 resident model、启动时固定 1–8 个 active requests 的 NInfer Engine/serve 路径

本文是一次有终点的实施计划，不是新的长期架构权威。全部治理项完成后，应把稳定不变量合并到
`engine-architecture.md`、`resource-scheduling-and-context-cache.md`、`paged-kv-cache.md` 和
`serving.md`，随后删除本文。

## 1. 目标和完成标准

本轮治理解决的是 CPU/Host 工作随请求长度、输出长度、媒体数量或 checkpoint 数量异常放大的问题，
以及这些问题目前无法从 serve 日志中直接观测的问题。目标不是用 catch、降级路径或启发式阈值掩盖慢
路径，而是消除错误的状态组织和算法复杂度。

完成后应同时满足：

1. 普通 decode round 的稳态 Host 工作只与 compact batch 大小和本轮新增状态有关，不扫描完整历史
   prompt、完整 KV frontier、完整 checkpoint catalog 或完整输出。
2. Prompt、媒体和 Responses 历史处理对其真实输入规模保持线性或 `O(N log N)`；不再通过逐次
   `erase/insert/find` 形成隐藏的二次复杂度。
3. 不随状态变化的验证在对象建立或 transaction reserve 时完成；每轮只验证本轮动态边界。
4. 普通decode不触发admission/catalog/placement重扫；只有waiting、lane、catalog或Program resource
   revision变化唤醒相应boundary policy。
5. 同步点只服务于可观察结果或资源生命周期。异步化必须同时给出 event、buffer、workspace 和取消路径
   的所有权，不能只删除 `synchronize()`。
6. Prefix reuse 始终按内容成立，不依赖 session/request identity。私有追加命中和跨会话 shared prefix
   命中都保留精确 token 比较作为最终裁决。
7. serve JSONL 直接报告 Host-owned 区域的实际 elapsed time，不以 `total - decode` 等差值猜测；在
   batch concurrency 大于 1 时仍能正确聚合。

以下不在本计划范围内：扩大到多 GPU、大规模抢占式 continuous batching、引入通用 profiler framework、
为旧的项目内日志 schema 保留兼容分支，以及把真实场景性能验收替代为合成测试。真实语料的
`ninfer serve` 验收仍由用户完成；实现侧负责构建、单元/契约测试、临时无模型 Host benchmark 和必要的局部
GPU 验证。

复杂度/性能 benchmark 是实施期一次性证据：临时源码和二进制放在构建目录，前后使用同一 workload，
测量完成即删除，不加入 CMake、CTest 或仓库维护面。长期只保留保护可观察语义或真实回归的测试。

## 2. 已确认的问题地图

下表来自当前代码路径审查。优先级表示治理顺序，不表示只有高优先级项目才需要完成。

| ID | 当前路径和根因 | 错误的规模关系 | 目标状态 | 优先级 |
|---|---|---|---|---|
| H1 | `logical_kv_store.h::commit_frontier` 在普通 decode 与 speculative commit 后扫描全部 mapped pages | 长输出累计约为 `O(output × prompt/pages + output²/pages)` | 只处理旧 tail、本轮新页和新 frontier；`O(delta pages)` | P0 |
| H2（已解决） | 旧`EngineCore::try_start_replica_transition`曾高频遍历catalog与checkpoint pages | 稳态每轮曾为`O(catalog + checkpoint pages)` | 周期性replica scan已删除；placement只属于具体Resume/Capture/Finish或admission pressure plan | — |
| H3 | `publish_runtime_stats` 在普通 unit 后复制/汇总结构状态；OutputSession 重复 token byte decode/线性查找；GDN fold 重验不可变几何 | 每轮固定 Host gap 偏大，部分工作随 catalog/token/layer 放大 | 增量计数、结构变更才发布；预解码 token record；prepared fold plan | P0 |
| H4 | `paged_kv_cache.cpp` 用有序 vector 的头部/中部 `erase`、`insert` 管理 free page | 连续申请/释放可达 `O(page_count²)` | free run/interval allocator；批量 materialize 与 page-table publish | P0 |
| H5 | `append_forced_tokens` 为 target-control suffix 复制完整 sequence ledger，并逐 row 执行和同步 | cap boundary 的 Host copy 随完整上下文增长；并发 rows 串行等待 | 在已 reserve 的 ledger 上事务式 append；建立可批量提交的 control ingress/event 生命周期 | P1 |
| A1 | Program联合pressure overlay仍可能逐页query/split/rebind | admission/pressure可达`O(pages²)` | dense generation marker或直接索引；按extent/run合并操作 | P1 |
| A2 | `has_active_reference` 每次遍历所有 active address/pages；`copy_from_host` 重复检查；Host extent 的 `node_at`/逐页释放 | pressure、copy、release 可达 `O(pages²)` | logical page active refcount；一次验证后的 prepared list；run cursor/release | P1 |
| C1 | Prefix shortlist 最终仍把所有 private/shared slot 送入 exact candidate 流程；capture marker 复制完整 prompt；静态 checkpoint 摘要重复生成 | candidate 数和 prompt/checkpoint 长度乘积放大 | 可查询任意 frontier 的 rolling content fingerprint；统一索引；共享 immutable backing；缓存静态摘要 | P1 |
| F1 | `encode_rendered_chat` 先全量 tokenize，再为每条 message/rewrite/cache boundary 重复 tokenize prefix | agent 追加历史接近 `O(turns × rendered bytes)` | 渲染前确定所需 marker；一次 boundary-aware tokenization；精确处理跨 boundary token | P1 |
| F2 | BPE merge 每轮扫描所有 pair 并从 vector 中间删除 | 单个长 word 为 `O(word²)` | integer symbol + linked adjacency + rank/leftmost priority queue；`O(N log N)` | P1 |
| F3 | Context capacity 在完整 Frontend 构造之后才拒绝；超长输入可先分配 token types、MRoPE 和媒体结构 | 错误输入先消耗与输入等比例甚至更高的内存 | tokenizer 最多产出 `max_context + 1` 后立即拒绝；`count_tokens` 保持独立精确路径 | P0 |
| V1 | Vision chunk 每次从首个 use 重找；visual overlap 每 chunk 重验完整 scatter；full cache hit 前仍构造全部历史 VisionControl | `O(media × chunks)` 或对已复用历史重复工作 | monotonic cursor；构造期验证；先选 reuse frontier，再只建 suffix control | P1 |
| V2 | Placeholder expansion 逐媒体 find/replace；媒体 count-only 路径执行 decode/resize/BF16 pack；媒体 acquisition 串行 | 随媒体数和 prompt 长度乘积放大，count 产生真实预处理开销 | 单次扫描+累计 delta；geometry/count-only 路径；有总 byte reservation 的有界并发 acquisition | P2 |
| V3 | 均匀 Vision segment 仍走通用 cu-seqlens 描述路径；RGB transient bytes 未进入 reservation | 多余 descriptor launch；并发预处理时资源账本不完整 | 使用既有 uniform overload；transient reservation 覆盖 decode/resize 生命周期 | P2 |
| S1 | 最终 MTP prefill 经 Host 搬运 4 bytes 后再整体 H2D；spec commit 存在可消除的第二次 wait；每个 prefill chunk 都同步 | 不必要的 D2H/H2D 和 Host blocking | final token 直接 D2D；event-owned speculative tail；在持久 ingress/workspace 生命周期建立后再 pipeline prefill | P2 |
| R1 | Responses store 在锁内重算/序列化整表与 ancestry；长 chain 累计二次；请求历史有大对象复制 | Responses chain 为 `O(chain²)`，锁持有时间扩大 | cached envelope bytes + context-node incremental live refcount；move/view ownership | P2 |
| R2 | ToolCall stream filter 对长尾 whitespace 反复处理；SSE 层存在重复 writable/select 探测 | 输出越长，consumer-side Host 工作非线性或重复 | incremental marker/trailing-whitespace state；只保留一个 transport readiness owner | P2 |
| R3 | 非 streaming request 仍经逐 token queue/lock/notify 发布，而 consumer 只在 terminal 读取聚合结果 | 每 token 不必要的锁和唤醒 | submission 时固定 AggregateOnly/Streaming consumer mode；AggregateOnly 只维护最终结果 | P1 |

这里的 “P0” 表示会直接破坏长上下文/长输出可用性或污染每轮 decode 的工作；“P1” 表示常见 agent、
cache/admission 或多媒体路径的结构性放大；“P2” 表示在 P0/P1 变为稳定基线后继续消除的同步和产品层
开销。

## 3. 治理不变量

### 3.1 复杂度预算

- 普通 decode round：`O(B + delta_pages)`，其中 `B <= 8`，`delta_pages` 是本轮真实跨越或新增的页。
  不允许出现对 total prompt tokens、generated tokens、全部 mapped pages、全部 cache entries 或全部
  response history 的扫描。
- Speculative round：允许与 draft window 成线性关系；commit 只处理 accepted delta 和必须折叠的动态
  state，不重验固定 layer/state geometry。
- Boundary：无相关事件时为`O(B)`；catalog/admission只由waiting、lane、catalog或Program resource revision
  变化触发；不存在周期性replica policy pass。
- Prompt preparation：`O(rendered bytes + tokens + media items)`，BPE 内部允许 `O(N log N)`；message
  boundary 数量不能再次乘以完整 rendered bytes。
- KV/Host extent 操作：以 run/extent 为单位；一个范围的申请、释放、copy、publish 不得退化为对有序
  vector 的逐页头删/中插。
- Vision：chunk 迭代使用 monotonic cursor；已由 reuse frontier 覆盖的 media 不构造 runtime control。
- Serving：每个新 response 的新增工作与本 response 和新增 ancestry edge 成线性关系，不重算整张 store。

### 3.2 状态与验证所有权

- 静态模型尺寸、layer range、scatter 单调性、checkpoint descriptor 和 output vocab bytes 在构造或 bind
  时验证/准备一次。
- Program seal生成opaque `ResourcePlan`；start只重验resource revision、capability和动态边界。失效后重新
  plan，不在execution中重复candidate推导。
- ResourceManager拥有logical candidate/retention policy；Program真实stores与allocators拥有physical
  occupancy、reservation与reclaim。EngineCore不通过“每轮再问一次”弥补缺少的event。
- RuntimeStats 的 token/round 数是 worker-owned monotonic counters；结构 gauges 只在结构变化后重建。

### 3.3 Cache 语义

- 查询键来自 token 内容和 frontier，不来自 request/session identity。
- Fingerprint 只用于 shortlist。命中必须经过 token 数、frontier 约束和精确 token 比较；哈希冲突不能改变
  语义。
- Private endpoint/turn closure/response replay/long anchor 与 shared stable prefix 使用同一内容索引协议，
  但保留不同 retention/replica policy。
- Capture marker 持有 immutable token backing 的 slice/frontier，不复制一份完整历史 prompt。

### 3.4 异步化约束

每个被移除的同步点都必须在设计中回答四个问题：

1. 哪个 CUDA event 表示数据可用；
2. event 完成前谁拥有 Host ingress、Device workspace 和 state selector；
3. cancel、terminal、exception 与 Engine 销毁如何回收；
4. 下一条可能跨 stream 或复用同一地址的路径在哪里 wait。

答不完整时保留同步点。普通 decode egress、speculative 首次 token 决策、exception cleanup 和析构同步
属于正确性边界，不作为“看见 synchronize 就删除”的治理对象。

## 4. Host 耗时的可观测契约

### 4.1 为什么不能使用差值估算

现有 `GenerationTimings.decode_seconds` 的边界不能覆盖 Program 返回后的 frontier commit、OutputSession
preview/commit、ResourceManager 更新、replica policy 和 stats publication。`total_seconds - decode_seconds`
又混合了排队、prefill、consumer 等时间。在 compact batch 下，同一段 wall time 同时暴露给多条请求，
因此把各 request 的差值求和还会重复计费。

新的统计同时提供两个互补视角：

- **Worker aggregate**：Engine worker 对每段时间只计一次，可跨请求、跨并发安全聚合，是判断 Engine
  实际 Host 成本的权威数据。
- **Request exposure**：一条请求在 active membership 中实际承受的完整 phase elapsed time。batch 中每个
  row 都获得完整 elapsed time，因此适合解释单请求延迟，但明确禁止跨请求求和。

任何字段都不以其他字段相减得到。排队、Host-owned active work、阻塞等待 GPU、Frontend preparation 和
HTTP consumer 分开报告。

### 4.2 时间语义

内部统一使用 `std::chrono::steady_clock`，以 `uint64_t` nanoseconds 累加，在 public result/JSON 边界才
转换为 seconds。这里测量的是 **Host-owned region 的 wall elapsed time**：它会包含该 worker 被 OS
抢占的时间，因为这同样表现为 GPU launch gap；它不是 thread CPU cycles。

`cudaStreamSynchronize`/completion-event wait 的 wall elapsed time 单列为 `device_wait`。它表示 Host
阻塞且 GPU unit 尚未完成，不计入 `host_active`。这一区分使以下两类问题可直接判别：

- `program_submit`/`program_post`/`engine_*` 增长：真实 Host orchestration 或算法复杂度问题；
- `device_wait` 增长而 Host phases 稳定：GPU execution、排队在 stream 上的工作或同步边界问题。

### 4.3 互斥的顶层 phase

Worker aggregate 使用以下互斥 phase；同一纳秒只属于其中一个 phase，`host_active` 可直接求和：

| Phase | 开始/结束边界 | 典型内容 |
|---|---|---|
| `engine_boundary` | worker 获得 execution ownership 到选定 execution unit | expiry/cancel、scheduler、membership、context progress 的 Host 部分 |
| `program_submit` | 进入 Program unit 到第一次必须等待该 unit 的 completion | ingress/page-table 准备、graph 选择、kernel/memcpy enqueue |
| `device_wait` | completion wait 调用前后 | CUDA stream/event blocking；不属于 `host_active` |
| `program_post` | wait 返回到 Program unit 返回 | D2H 小结果读取、frontier 增量 commit、Program state bookkeeping |
| `engine_commit_output` | Program 返回到 token/state/output transaction 完成 | preview、Program 调用之外的 commit orchestration、ResourceManager apply、result publication |
| `engine_maintenance` | execution transaction 后到 boundary 完成 | 结构 stats publication、必要的 cleanup |

Program 中一个 unit 若有多段 submit/wait/post（例如 speculative commit tail），各段分别累加到同名
phase；仍保持互斥。Program API 返回该 unit 的结构化 observation，EngineCore 负责把它并入全局计数并
向 batch rows 记 exposure。EngineCore 调用 Program execution/commit 时暂停外层 phase，Program 返回后再
恢复 `engine_commit_output`，因此 Program 内部时间不会被双计。不能让 serve 层根据总 wall time 反推
Program 内部边界。

为了定位低频 policy，额外记录以下 **subset counters**：`admission_policy`、`context_progress`、
`stats_publication`。它们明确标为 `detail_subset_seconds`，包含在上表某个顶层 phase 中，
不再次加入 `host_active`。这些 timer 只在对应慢路径真正执行时启动；稳态 clean boundary 不承担额外
clock call。M0 刚落地而 M1 尚未治理时，某个 helper 可能仍被每轮调用；timer 只跟随实际 invocation，
M1 删除高频 invocation 后自然退出稳态路径。

### 4.4 计数和归一化

Engine 内部增加 worker-owned `HostWorkCounters`：

```text
top-level phase elapsed_ns (monotonic, mutually exclusive)
detail subset elapsed_ns + invocation_count
decode_host_active_ns
decode_device_wait_ns
prefill_host_active_ns
prefill_device_wait_ns
control_host_active_ns
control_device_wait_ns
decode_rounds / decode_row_rounds / prefill_units / control_units
```

`RuntimeStats` 暴露这些 monotonic integer counters。serve throughput reporter 只做相邻 snapshot 的
monotonic delta，不重新遍历 Engine 状态。以下归一化值由 delta 直接计算：

- `decode_host_microseconds_per_round = decode_host_active_ns / decode_rounds / 1000`；
- `decode_host_microseconds_per_row_round = decode_host_active_ns / decode_row_rounds / 1000`；
- `decode_device_wait_microseconds_per_round`；
- 每个 detail policy 的 `microseconds_per_invocation`。

混合了 prefill/control 的 interval 不能拿总 Host 时间除以 decode rounds。分母为零时 JSON 写 `null`，
不写 `0` 或无穷值。

每段顶层 phase 还带一个 execution class。Program phase 由其 unit 直接标为 decode、prefill 或 control；
unit 后的 commit/maintenance 继承该 unit。选中 decode 前发生的 boundary 工作（包括一次失败但允许继续
decode 的 admission check）计入 decode，使 `decode_host_microseconds_per_round` 反映真实 launch gap；
实际选中 materialization/context/replica transaction 的 boundary 计入 control。condition-variable idle wait
不属于任何 Host phase。

每条 RequestRecord 只保存 exposure 累加和及 unit counts：

- request 安装为 active owner 后，直到 terminal/abort boundary，期间所有 worker phase 和 device wait 都
  全额计入 exposure；即使本轮是另一条请求的 prefill/control unit，当前 active request 也确实被该 unit
  延迟；
- request 自己参与的 decode batch 另记 decode-class subset；batch 中每个 row 都获得本轮完整 elapsed，
  不按 batch size 均摊；
- admission 前的 candidate 尚未 active，其等待（包括自身 admission transaction）由 `queue_wait_seconds`
  表达；admission 的全局 Host 成本由 worker aggregate 的 detail subset 表达；
- `queue_wait_seconds` 从 submit 到成功安装 active owner 单独记录；
- preparation/media acquisition 沿用现有 preparation 字段；HTTP sink/transport 时间不混入 Engine Host。

因此 request exposure 反映“这条请求承受了多少延迟”，不是“把共享 CPU 成本按 row 均摊多少”。

### 4.5 serve JSONL 与 console 输出

实现时将当前 request-log schema 递增一次，并同步 `tests/test_request_log.cpp`、
`tools/bench/run_serve_corpus.py` 与 `docs/serving.md`。项目内 schema 不保留双写或旧字段 fallback。
本文编写时当前 schema 为 13；若实施前已有别的 schema 变更，则从届时当前版本递增，而不是硬编码兼容
13。

`request_done` 增加：

```json
{
  "engine_timing": {
    "queue_wait_seconds": 0.001234,
    "host_exposed_seconds": {
      "engine_boundary": 0.000321,
      "program_submit": 0.001112,
      "program_post": 0.000842,
      "engine_commit_output": 0.001704,
      "engine_maintenance": 0.000103,
      "total": 0.004082
    },
    "device_wait_exposed_seconds": 0.812345,
    "decode": {
      "host_exposed_seconds": 0.003462,
      "device_wait_exposed_seconds": 0.801220,
      "rounds": 127
    },
    "units": {
      "prefill": 4,
      "control": 0
    }
  }
}
```

字段名和 schema 文档明确规定 `exposed` 值不可跨 request 求和，无需为这一固定事实在每行重复写一个
布尔值。`host_exposed_seconds.total` 必须等于五个互斥 Host phase 之和，`device_wait` 不在其中；
`decode` 是其中按 execution class 统计的 subset，供单请求 decode 归一化使用。

周期 `throughput` 增加可聚合的 worker delta：

```json
{
  "host_work": {
    "elapsed_seconds": {
      "engine_boundary": 0.003201,
      "program_submit": 0.004827,
      "program_post": 0.002119,
      "engine_commit_output": 0.006443,
      "engine_maintenance": 0.000781,
      "total": 0.017371
    },
    "device_wait_seconds": 4.911002,
    "detail_subset_seconds": {
      "admission_policy": 0.000412,
      "context_progress": 0.000000,
      "stats_publication": 0.000133
    },
    "detail_invocations": {
      "admission_policy": 2,
      "context_progress": 0,
      "stats_publication": 1
    },
    "decode_host_microseconds_per_round": 31.42,
    "decode_host_microseconds_per_row_round": 12.77,
    "decode_device_wait_microseconds_per_round": 3817.05
  }
}
```

这些数字是示意，不是目标基线。Console 只保留最能立即判断回归的一行摘要，例如：

```text
host=17.37ms decode-host=31.4us/round wait=3817.1us/round boundary=3.20ms maintenance=0.78ms
```

单请求完成行增加 `host=<ms>`、`decode-host=<us>/round` 和 `wait=<us>/round`；完整 phase 只写 JSONL，
避免 stderr 本身成为高频开销。日志仍只在 request terminal 或 stats interval 输出，不产生
per-token/per-round 日志。

### 4.6 插桩位置与开销预算

插桩归 Engine/runtime 所有，serve 只翻译已完成的 measurement：

- `include/ninfer/types.h`：定义 public request exposure 和 RuntimeStats monotonic counters；
- `src/runtime/engine/request_record.h`：保存 request exposure；
- `src/runtime/engine/engine_core.h`：boundary/commit/maintenance 计时、batch exposure 和全局累计；
- Qwen family Program execution result/commit result：返回 submit/wait/post observation；两个 target package
  通过同一 family runtime 获得相同语义，不加 target runtime 分支；
- `src/core/nvtx.h` 与对应调用点：复用已有静态 registered names，并为缺失的 boundary/post/commit phase
  增加静态 range；NVTX 边界与计时边界相同，不构造 dynamic range string；
- `src/serve/generation_service.*`：原样复制 Engine observation；
- `src/serve/request_log.*`：schema/JSON/console；
- `tools/bench/run_serve_corpus.py`：解析 raw fields，默认汇总 worker aggregate，单请求只展示 exposure
  distribution。

普通 decode round 最多执行固定数量的 coarse `steady_clock::now()`，不在 token、page、layer、media 或
candidate 内层循环计时。detail timer 仅在慢路径执行。worker-owned 累加不使用 atomic、mutex、heap
allocation 或 string formatting；RuntimeStats snapshot 只是整数复制。

临时无模型 Host microbenchmark 需测量同样 phase-switch 序列，验收条件为：

- profiler 未附加时，包含 coarse NVTX push/pop 在内，每个普通 decode round 的插桩增量不超过
  `0.5 us`；
- 热路径零 allocation、零 lock；
- 关闭 JSONL 时仍累计 RuntimeStats，但不做 JSON/string 构造；
- 开启 JSONL 的 serialization 只发生在既有 terminal/interval reporting path 上，不占 Engine worker。

如果达不到该预算，应减少 detail phase，而不是采样顶层 phase 或加入运行时开关。顶层数据必须始终可比。

## 5. 分阶段实施

### M0：建立可观测基线

实施第 4 节的 Host timing contract，先不改变 scheduling/cache policy。该阶段的目的不是用旧实现数据
决定是否修复已确认的复杂度错误，而是让后续每项治理都能在相同口径下比较。

交付：

1. Engine/Program structured timing 和 RuntimeStats monotonic counters；
2. request exposure 与 throughput worker aggregate；
3. 与相同 phase 边界对齐的静态 NVTX ranges；
4. request-log schema、console、serve corpus parser、文档和受影响的 request-log 契约测试同步更新；
5. Host instrumentation 临时 microbenchmark 的测量结论；
6. 对现有 `prepare/prefill/decode/total` 边界补充准确注释，不改变它们来伪装 Host 指标。

验收：phase 总和在整数 nanoseconds 上精确相等；subset 不大于其 parent；snapshot delta 单调；batch=2
测试证明 aggregate 只计一次而两个 request exposure 各计完整 elapsed；零分母输出 `null`。

### M1：先消除长上下文/稳态 decode 的 P0 问题

#### M1.1 早期 context guard

- Frontend full encode 使用最多 `max_context + 1` token 的 bounded mode；发现超限后在 boundary、MRoPE、
  media runtime structure 构造前返回 ContextLengthExceeded。
- `prepare_tokens` 在进入 target Frontend 前检查 token 数；`count_tokens` 是精确计数 API，保留独立
  unbounded count 路径。
- 不捕获 `bad_alloc` 并改写错误；目标是使合法的 capacity rejection 在大分配之前发生。

验收：超长 text/token 请求不建立 per-token auxiliary arrays；恰好 max_context 与 max_context+1 的边界
行为正确；tokenizer 的正常输出与现状逐 token 相同。

#### M1.2 增量 frontier commit

- `KVAddressSpace` 记录 committed frontier 对应的 tail page/coverage。
- frontier 前进时只处理旧 tail coverage、新跨越 pages 和新 tail；frontier 截断走明确的 truncate
  transaction，不复用 forward-commit 路径。
- 在所有新 page coverage 成功提交后一次发布新 frontier，异常时保留旧 frontier，避免半发布。
- Text 与 backend KV 共用同一状态机语义，各自保持独立 address space。

验收：测试注入 page-touch counter；frontier 每次增加 1 token 时，touch 次数只在跨页时增长，不随旧
frontier 增长；普通和 speculative accepted delta 得到相同最终 coverage。

#### M1.3 删除周期性replica scan，接入ResourcePlan（已完成）

- 删除独立`try_start_replica_transition`热路径；placement只由具体Resume/Capture/Finish或admission pressure
  choice触发。
- Program维护唯一`resource_revision`。只有stable topology、global free capacity或allocator结构变化推进；
  active mapped/reserved转换、普通decode、output和stats不推进。
- ResourceManager在相关逻辑事件发生时建立choice；Program一次完成joint post-state、stage peaks与exact
  allocator preflight并seal `ResourcePlan`。
- Start前revision变化安全拒绝且无物理副作用；start后由一个`RunningTransaction`到达commit或abort，不缓存
  第二份physical snapshot。

验收：稳态长decode中replica catalog/page visits为0；无Resume/Capture/Finish/cleanup时placement visits为0；
stale plan不产生物理副作用。

#### M1.4 每轮固定 Host 工作

- RuntimeStats token/round/state-transfer数增量更新；结构gauge只在Program resource revision或明确采样
  boundary读取，不为统计维护第二份physical ledger。
- Tokenizer/Frontend 在模型加载时建立一份 immutable vocab token-bytes/flags table，所有 OutputSession
  只引用它，不能为每条请求复制整份词表。每个 generated token 只建立一个 immutable decoded record，
  preview、stop matcher 和 commit 共享。
- stop strings 编译为 streaming matcher，处理新增 bytes，不从头扫描完整输出。
- GDN speculative fold 在 Program 构造/sequence reserve 时建立 prepared fold plan；commit 只检查 row、
  accepted span 和动态 state selector。
- `append_forced_tokens` 直接在 admission 已预留 capacity 的 sequence ledger 上做事务式 suffix append，
  失败时回滚 size 或清理 sequence；不为一次 control suffix 复制完整历史 ledger。
- request submission 固定 consumer mode。AggregateOnly request 不建立逐 token queue、不做 per-token
  notify；Streaming request 保持增量 publication 和 backpressure 语义。
- pending FIFO 使用 `waiting_count` 和 earliest-deadline owner，不为 empty check 复制 snapshot，也不在
  每个 boundary 扫描无过期可能的全部 pending records。

验收：普通 decode 的 `program_post`、`engine_commit_output` 和 `engine_maintenance` us/round 不随已生成
长度形成正斜率；长 stop-prefix 输入与输出长度保持线性。

#### M1.5 KV free-run allocator 与批量 publish

- 用有序 free intervals/runs 取代逐页有序 vector；连续 preferred run 直接 split/consume，释放时与相邻
  run 合并。
- 增加 batch materialize API，一次返回 validated page list/run list；page table entries 连续写入后执行
  一次或少数合并 publish，而不是每页 4-byte transfer。
- allocator 不改变 logical page identity、generation 或 CUDA Graph 地址稳定性契约。

验收：申请/释放 P 个连续 pages 的结构操作为 `O(number of runs log runs)`；随机碎片测试与朴素 bitmap
oracle 的 free/allocated 集合完全一致；page-table 最终内容精确一致。

### M2：线性化 admission、pressure 与 prefix candidate

#### M2.1 page/resource scratch

- Program对完整choice使用page-id可直接索引的generation marker/dense scratch，一次pass构造联合post-state；
  scratch是可重建加速结构，不是ResourceManager physical mirror。
- logical page 维护 active-address refcount，在 address bind/rebind/truncate/release transaction 中增减；
  `has_active_reference` 为 `O(1)`。
- `copy_from_host` 在 reserve 时生成 generation-bound validated page/run list；execution 不再逐项向前
  搜索 duplicate。
- Host extent store 维护 linked/run cursor，按 contiguous run `release_unreferenced`；pressure 以 extent
  run 合并 split/rebind。

验收：用operation counter证明projection、active reference query、copy validation和release分别只线性访问
输入；abort不留下半Active target，并由`ResourceResult`准确报告可能已提交的victim changes。

#### M2.2 内容驱动的 prefix index

- 为 incoming prompt 建立可在任意 catalog frontier 查询的 rolling token fingerprint。
- endpoint、turn closure、response replay、long anchor 与 shared stable prefix 进入统一的
  `(frontier, fingerprint, semantic boundary kind)` 索引；retention 与 replica policy 仍由各 kind owner
  管理。
- shortlist 后才进行精确 token compare。没有命中的 fingerprint bucket 不追加全部 slots 作为
  “fallback candidates”；root 是确定的最终候选。
- checkpoint 的静态 token digest/resource summary 在 checkpoint 创建或更新时缓存。
- active capture marker 持有 immutable prompt backing + frontier slice，不复制完整 token vector。

验收：同一会话的简单 prompt append 无 identity 仍完美命中旧 frontier；不同会话的 shared system
prefix 命中；构造 hash collision 仍由精确比较拒绝；无关 catalog 增长不增加 exact comparison 数。

### M3：Frontend 和 tokenizer 线性化

#### M3.1 一次 boundary-aware tokenization

- 渲染阶段只记录 rewrite/cache/message 的 byte marker；每个 marker 在线性 tokenization 结果上做常数级
  frontier 映射，不为任何 marker 生成独立 token prefix。
- Tokenizer 提供一次 encode 同时返回 marker 到 token frontier 的结果。跨 marker 的 tokenizer token
  必须按完整文本语义决定，不能简单把每段独立 tokenize 后拼接。
- 旧的逐 prefix encode 只可留在测试 oracle，生产路径完全删除，不保留双轨 fallback。

验收：覆盖 token 跨 message/cache marker 的 adversarial 文本；完整 token 序列和结构性 exact frontier
精确一致；跨 token/cache marker 的 stable frontier 只保留 normalization/pre-tokenization 已完成的前缀，
不依赖 marker 后字节；增加 turn 数不会增加 full-tokenize 调用次数。

#### M3.2 BPE 与 placeholder

- BPE 使用 integer symbol nodes、双向 adjacency、按 `(rank, leftmost position)` 排序的 priority queue；
  stale queue entry 通过 node generation 丢弃。
- 保留当前 merge rank 和 leftmost tie 语义，输出由独立现有 tokenizer fixture 精确比较。
- 媒体 placeholder 通过一次 rendered-text scan 生成 segments 和 source→rendered cumulative delta map；
  rewrite/cache boundaries 一次映射，不逐媒体修改全部后续 offset。

验收：临时长 single-word benchmark 的 operation slope 为 `O(N log N)`；普通/特殊 token fixture 精确一致；
媒体数增长时 rendered text 只完整扫描一次。

### M4：Vision 与媒体路径

- `VisionPrefillSession` 保存下一个 visual-use cursor；chunk 只向前推进。
- scatter sorted/unique、grid 和固定 shape 在 `VisionControl` 构造时验证；`shifted_visual_overlap` 运行期
  只做 lower_bound/范围交集。
- RequestPlan 先选择 reuse source/frontier，再为未复用 suffix 构造 VisionControl；grid-intrinsic metadata
  缓存在 prepared media 中。
- count-only 根据 container metadata/geometry 和 tokenizer placeholder 规则计算，不 decode pixels、resize、
  BF16 pack 或写 media cache。
- acquisition 使用有界 worker pool，并在启动任务前 reservation aggregate compressed/raw/transient bytes；
  digest 在 acquisition 流中计算。Responses continuation 保存 acquired immutable media/digest，避免重复获取。
- uniform Vision segment 直接调用既有 uniform attention overload，不构造通用 cu-seqlens descriptors。

验收：chunk 数增长不重访旧 visual uses；full prefix hit 的历史媒体不构造 suffix runtime control；count-only
不产生 patch buffer/cache mutation；并发媒体峰值不超过 reservation；Vision 数值由现有模型 oracle/fixture
保持。

### M5：有生命周期证明的异步化

按风险由低到高实施：

1. 最终 MTP prefill token 从已有 Device output 直接 D2D 到 `mtp_ids` 最后一列，删除 4-byte D2H 和整行
   H2D；
2. target-control membership 使用启动时固定的 per-row ingress slices，在同一 stream 上 enqueue 全部 row
   的 forced-token work 后只等待一次 completion；不复用 event 完成前仍被 DMA 读取的 Host ingress；
3. speculative commit 在第一次必要 decode egress wait 后 enqueue fold，并返回 completion event；只有
   terminal/cancel/fork/cross-stream reuse/deferred recycle 等消费者等待；
4. prefill chunk pipeline 先建立 pinned/persistent ingress 和 event-owned workspace lease，再允许下一 Host
   chunk preparation 与当前 GPU execution overlap。

每步独立提交和验证，不把三种生命周期一次混合。验收除数值/state oracle 外，还要覆盖 cancel、exception、
terminal、fork/reuse 与 Engine destruction；任何地址在 event 完成前不得重用。

### M6：serve/Responses consumer 路径

- ResponseStore node 保存一次 serialization/envelope bytes 和 direct parent edge；put/erase 用 context-node
  live refcount 增量维护 byte/record capacity，锁内不遍历完整 ancestry/table。
- HTTP translation 对已拥有的大 message/media/tool values 使用 move 或 immutable view，消除 history 的
  多层复制。
- ToolCallStreamFilter 保存 incremental marker state 和 trailing-whitespace span，不重新扫描历史 buffer。
- readiness 由 cpp-httplib/transport 单一 owner 管理，删除 NInfer 层重复 select/writable probe。
- request log 可增加独立 `consumer_publish_seconds`，只测 sink/filter/protocol callback；它不进入
  `engine_timing.host_exposed_seconds`。

验收：response chain 每新增一项只触碰新 node 和直接 edge；tool whitespace work 与新增 bytes 线性；
stream/non-stream response body 和外部协议 schema 保持一致。

## 6. 实施顺序和变更边界

剩余工作推荐按以下可审查单元落地：

1. M0 Host measurement contract；
2. M1.1 context guard；
3. M1.2 frontier commit；
4. M1.4 + M1.5 每轮固定成本和 allocator；
5. M2 resource/cache candidate；
6. M3 Frontend/tokenizer；
7. M4 Vision/media；
8. M5 async；
9. M6 serve consumer。

每个单元都必须删除被替代的项目内生产路径，不能留下 `legacy`、catch fallback、双写 schema 或通过配置
选择旧算法。若一个阶段暴露新的问题，只在它阻塞本阶段正确性或使测量失真时并入；其他发现进入本文
的问题地图并单独排序。

M0 之后每个单元的交付说明统一报告：

- 改变了哪个复杂度/所有权不变量；
- 对应 Host phase 和 operation counter 的前后变化；
- 已运行的构建、语义测试和临时无模型 benchmark 结论；
- 需要用户用真实 `ninfer serve` 验收的 workload 与预期日志信号。

## 7. 验收矩阵

| 场景 | 正确性判据 | Host/复杂度判据 | 主要日志信号 |
|---|---|---|---|
| 64K prompt 后长普通 decode | token/state/cache frontier 正确 | frontier page touches 只随跨页增长；steady decode host/round 无随输出长度上升趋势 | `program_post`, `decode_host_us/round` |
| MTP/spec long decode | accepted/commit/GDN state 与 oracle 一致 | fold validation 与固定 layer 数无每轮重扫；无第二个无必要 wait | `program_post`, `device_wait` |
| agent prompt 连续追加 | 每轮复用前缀精确正确 | exact candidates 与命中 bucket 大小相关，不与 catalog 总量相关 | reuse path/tokens, `admission_policy` |
| 跨会话相同 system prompt | shared prefix 精确命中 | 不依赖 request/session identity | shared selection/reused tokens |
| 长多轮 chat preparation | token/frontier 精确一致 | 一次 full tokenize；工作近似随 rendered bytes 线性 | preparation tokenize seconds + benchmark operation count |
| 大量/长 word | tokenizer 输出精确一致 | BPE `O(N log N)`，无 vector 中删 | tokenizer benchmark |
| 多媒体、多 chunk | Vision 输出/position 正确 | cursor 不回扫；full reuse 不构造旧 media control | prepare/media stats, `program_submit` |
| admission/pressure/host restore | joint post-state、stage peak和terminal result正确 | page/extent pass线性；普通decode placement visits为0 | policy subset seconds/invocations |
| C=2–8 compact batch | 每 row 输出和取消语义正确 | worker aggregate 一段只计一次；request exposure 各计完整延迟 | throughput aggregate vs request exposure |
| Responses 长 chain/stream | 外部 schema/body 一致 | put/filter work 随新增内容线性 | consumer timing；Engine timing不被混入 |

性能回归判断以 raw JSONL 的 worker aggregate 为主；stderr rounded summary 只用于现场观察。临时墙钟
benchmark 不替代语义测试，局部复杂度证据也不替代用户真实语料的端到端验收；这些 benchmark 不作为
长期 target 保留。

## 8. 计划完成与文档收口

满足以下条件后本计划结束：

1. P0/P1 项全部落地，P2 项完成或有基于新 Host 数据的明确保留理由；
2. request_done 与 throughput 都具备第 4 节语义，serve corpus parser 使用 worker aggregate；
3. 长 prompt/长输出/多媒体/Responses chain 的 operation-slope 测试通过；
4. 用户的真实 `ninfer serve` 场景没有 Host phase 随历史长度异常增长，cache reuse 语义保持；
5. 稳定复杂度、cache、transaction 和日志契约已合并进对应 active maintainer/public docs；
6. 本临时计划被删除，不保留平行的“最终版”或兼容说明。
