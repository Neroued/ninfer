# 产品入口、工具链与分布式执行收尾清单

更新日期：2026-10-28。保留原问题编号，逐项推进。

当前剩余：**9 待实现，10 待定性**。1—8 已完成。
本轮另定位了启动 headroom 的规划／分配额度混用，见第 7 项验收备注。
状态依据当前源码和各项实测记录。

## 1. Perplexity 接入公共启动配置与 deployment

- [x] 完成入口接入。

入口：[apps/perplexity/main.cpp](../apps/perplexity/main.cpp)。

已使用公共 `EntryPoint::Perplexity`、TOML／CLI 解析及 deployment；生成专用配置按入口隔离。
PPL 保留自己的语料、评分窗口与 stride；窗口决定未显式配置的 Engine context 默认值。
报告使用公共启动资源统计，记录实际各 rank 与总量，窗口规划和 NLL 聚合数学不变。

验证：5090 Qwen3.5、Spark Qwen4Exp 单机／双机和 GLM 双机均通过真实 PPL 命令评分，
覆盖 BF16／FP8 KV、多窗口与分块执行。配置覆盖、目标范围、NLL 聚合、各 rank 资源与总量
已核对；记录位于 `.agent-tmp/ppl-integration/`。

## 2. 串行 serving 基准对齐生产日志

- [x] 修复 Vision 字段、计时有效性与 Host 阶段对账。

入口：[run_serve_corpus.py](../tools/bench/run_serve_corpus.py)。

任务：读取 `timings_seconds.vision_stream`，结合
`execution.max_rank_gpu_intervals.vision_stream.count` 区分未测量与测得零值；
Host 阶段统计和求和补齐 `executor_flush`，与生产端的 total 保持一致。

完成条件：真实生产日志可解析；关闭 GPU timing 时不报告虚假的零耗时；
`executor_flush` 非零时，阶段总量核对和结果输出正确。

## 3. 串行 serving 基准修正 TTFT 口径

- [x] 直接消费服务端首 token 延迟。

入口：[run_serve_corpus.py](../tools/bench/run_serve_corpus.py)。

任务：使用生产日志的 `timings_seconds.ttft`，替换 prepare、Vision、prefill 相加的公式；
对齐受影响的结果字段和指标说明，区分 server TTFT 与客户端网络端到端 TTFT。

完成条件：报告中的 server TTFT 与生产日志一致，排队和阶段重叠不会被错误求和。

## 4. 并发基准与 TTFT 诊断保留启动资源

- [x] 迁移启动资源消费者。

入口：[run_serve_concurrency.py](../tools/bench/run_serve_concurrency.py)、
[ttft/diagnostics.py](../tools/bench/ttft/diagnostics.py)。

任务：从 `server_start.resources` 读取 `ranks` 和 `total_allocations`；保留 rank 身份、
资源明细与汇总。缺失或无法解析时明确标记不可用，替换旧 `memory`／`startup_memory` 消费。

完成条件：单机、双机报告均保留实际启动资源；缺失数据可识别，不静默生成正常空报告。

复查：2—4 的消费者仍与生产端一致；共用提取函数为
[serve_log.py](../tools/bench/serve_log.py)。现有 `tests/test_serve_corpus.py` 的 25 项检查通过，
覆盖未测量／实测零值、server TTFT、Host 阶段求和及单／双 rank 资源。
本次还用刚关闭的双 Spark 服务启动日志确认：资源提取结果为 `available`，保留两 rank 和总量。

## 5. Profiler 与峰值重置使用完整执行边界

- [x] 接通实际执行 rank，并统一测量窗口的生命周期。

入口：[ninfer_bench.cpp](../bench/inference/ninfer_bench.cpp)、
[EngineCore](../src/runtime/engine/engine_core.h)、
[Executor](../src/runtime/executor/executor.h)。

任务：将 profiler 开始／结束交给实际 CUDA context 所属执行路径，包含单机和所有
分布式 rank。统一审查 `reset_measurement_peaks()`：请求结果返回后，仍需处理尾部执行退役、
Session 待提交的 residency release 和相关 context work，不能直接以结果返回作为空闲证明，
也不能把正常收尾当作非法 busy。

`begin_measurement`／`end_measurement` 与独立峰值重置均由 Engine 执行线程处理。
`Measurement`／`MeasurementComplete` 按命令退役边界下发并汇总全 rank 观测，包含配对和失败清理。

完成条件：5090 与双 Spark 的各 rank 捕获只包含正式测量请求，覆盖最终提交，排除加载、
Graph 准备和预热；结果返回后的尾部工作能正常收尾；失败后不遗留开启的测量状态。

复查：bench 已调用公共 `begin_measurement()`／`end_measurement()`，不再自行控制 CUDA
profiler；Engine 执行线程等待请求及尾部 context work 完成，worker 等待命令退役后再执行
测量命令，并向协调端返回全 rank 结果。异常路径保留 Cancel 清理。

5090 短 bench 已通过预热和连续两组测量；2026-10-25～26 的 GLM 双 Spark 实测也已完成
预热、三轮正式 bench 和两个 worker 的 profiler 采集，覆盖 P64K 与 D/B1、B4、B8。
原来的“等待双机验收”已过时。记录见 `.agent-tmp/measurement-boundary/` 与
`.agent-tmp/glm5next-perf-20261025/`；D trace 按时长截取，P64K 采集完整测量请求。

## 6. 分布式 worker 启动进度贯通

- [x] 将已有启动阶段传回协调端并按来源展示。

`StartupEvent` 携带 rank／node、准备次数与阶段实例；耗时由产生事件的 rank 计算。
Qwen4Exp 与 GLM 的 worker observer 已接入加载、通信预热、资源准备、Graph 校准／捕获及 Ready。

启动诊断与运行时命令进度分开。诊断队列有独立容量，中间进度合并；阶段回复携带最终快照。
Renderer 按来源和阶段实例维护状态，忽略重复／过期事件。Ready 后停止转发，日志回调异常不影响启动。

验证：本机与 Spark 的日志／协议测试通过；双机执行用例覆盖抛异常的 observer。
Qwen4Exp、GLM 均完成双机 Graph 启动与请求，保留各 rank 的独立阶段耗时；GLM 还检查了终端交互展示。
记录位于 `.agent-tmp/startup-closeout/`。

## 7. 分布式启动失败保留原始原因

- [x] 修正失败消息、进程退出与连接结束的处理次序。

Transport 按 peer 交付完整消息及连接结束事件；写端失败后继续接收，单个 peer 失败不停止其他连接。
ExecutionGroup 统一消费诊断、业务消息与进程状态，优先保存完整 `Failed`，补充 rank／node、阶段及准备次数。
进程启动失败保留有界日志尾部；清理错误不覆盖原始原因。模型准备与运行时调用方不再各自检查退出状态。

验证：协议测试覆盖分片消息、紧随其后的 EOF、截断消息、其他 peer 继续接收、诊断额度耗尽及正常退出。
真实双机缺失模型路径完整保留远端路径和 `No such file or directory`；执行失败与缺失 worker 用例保留原始原因，
没有遗留子进程。RTX 5090 和两台 Spark 均完成完整构建。

验收备注：GLM 的日常 `auto` 配置曾在分配前被容量复核拒绝，同配置后续启动及请求成功。
两机启动前均清过文件缓存。现有 `StartupMemoryBudget::available()` 已扣除 headroom，
worker 在实际分配前仍用该规划额度复核；正常内存变化超过 Context 按块取整留下的余量时，
即使尚有 headroom 也会提前失败。需要分别提供规划额度与实际分配额度，后者允许消耗规划 headroom，
并继续由最终预算检查保护总额度和 Host 保留容量；该预算修正尚未实施。

## 8. KV 容量曲线的旧差分外推

- [x] 原根因已由统一 Context 池重构消除。

当前 [KV page 布局](../src/core/paged_kv_geometry.cpp)将各层、各 plane 及其对齐纳入完整
page record；容量规划直接使用真实 `page_bytes` 和统一池几何，最终资源构造仍保留相等检查。
原来的相邻页数差分公式已删除，不再安排该旧公式的修复。

实测补充：2026-10-28 双 Spark 的 Qwen4Exp TP2、FP8 KV、262144 context、MTP3、
Graph／Vision 配置已成功启动并完成请求，Context 池自动规划与最终资源构造正常。

## 9. Vision 前缀索引的摘要查询边界

- [ ] 分离身份查询与 checkpoint 捕获／恢复合法性。

入口：[session/request.cpp](../src/models/qwen4_exp/session/request.cpp)、
[PrefixIndex](../src/runtime/engine/context_cache/prefix_index.h)。

现状：Qwen4Exp 的 `RequestBasePlan::prefix_shortlist_key()` 仍调用 `can_checkpoint()`。
压缩索引的中间分支可以位于图像内部，因此后面的合法 checkpoint 仍可能从候选中遗漏。
Qwen3.5 与 GLM 当前的摘要查询没有这个限制；问题范围是 Qwen4Exp 的查询合同实现。

任务：合法输入范围内的摘要查询不受 `can_checkpoint()` 限制，使图像内部的索引分支也能
完成身份比较；捕获和恢复继续检查各自的合法边界，保持现有容量和保留策略。

完成条件：Vision 续接可找到索引分支之后的有效历史；换图不会误复用；图像内部非法
checkpoint 捕获和恢复仍被拒绝。

## 10. 双 Spark MTP grammar 的 schema 切换复用断言

- [ ] 固定前提并完成定性。

参考：[现有 grammar 用例](../tests/models/qwen3_5/test_engine_grammar_real.cpp)。

现状：旧双机失败日志仍保存在 `.50` 的 `.agent-tmp/capability-fixes/dual-grammar-mtp.log`，
错误为 `schema switch did not exercise prefix reuse`。当前用例仍检查
`reused_prompt_tokens > 0`，但该序列中历史是否应当持续保留尚无定性证据。
后来的普通 grammar／工具切换验证不能替代这一序列的缓存前提检查。

任务：固定失败序列、部署、缓存容量和保留前提，分别核对 schema 输出、历史是否仍保留、
索引候选与实际来源选择；明确该场景要求复用的合同依据，再决定是否存在引擎缺陷。
保留原始负载，不通过更换提示词或增加预热追求断言通过。

完成条件：给出有证据的定性及所属模块；确认违约后再确定修复范围。未命中本身不能直接
定性为约束解码或 MTP 数值错误。

## 验证范围

优先复用现有用例和真实产品入口，新增测试只覆盖具体风险。上述任务不包含八组测试的
机械迁移、重复测试入口、测试框架改造或新的性能对比活动。
