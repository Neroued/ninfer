# 产品入口、工具链与分布式执行收尾清单

核对基点：`dev / 1fcc6dc5`，2026-10-23。保留原问题编号，逐项推进。
第 8 项的旧根因已被统一 Context 池重构消除；第 10 项先定性，再决定是否修复。

## 1. Perplexity 接入公共启动配置与 deployment

- [ ] 完成入口接入。

入口：[apps/perplexity/main.cpp](../apps/perplexity/main.cpp)。

任务：复用公共启动配置，将 deployment 和引擎资源配置传给公共 Engine；保留 PPL
自身的语料、窗口、stride、NLL 聚合与输出逻辑。报告使用公共启动资源统计，包含实际
rank 信息与总量。

完成条件：当前受支持的单机、双 Spark 组合能通过真实 PPL 命令评分；配置覆盖正确；
窗口覆盖、NLL 聚合及实际 rank／资源信息正确。

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

5090 短 bench 已通过预热和连续两组测量；双 Spark 实际捕获留到随后性能分析时验收。

## 6. 分布式 worker 启动进度贯通

- [ ] 将已有启动阶段传回协调端并按来源展示。

入口：[rank preparation](../src/models/qwen4_exp/parallel/preparation.cpp)、
[StartupEvent](../include/ninfer/types.h)、
[StartupLogRenderer](../src/product/logging/startup_log.cpp)。

任务：为事件明确 rank／node 来源，贯通 worker observer 到协调端；renderer 按来源和 phase
分别维护进度、耗时及失败状态。转发有界，日志回调异常沿用诊断发布边界的隔离规则。

完成条件：能辨认各 rank 正在执行和已完成的阶段；耗时不串用；进度与日志失败不改变启动结果。

## 7. 分布式启动失败保留原始原因

- [ ] 修正失败消息与进程退出的处理次序。

入口：[bootstrap.cpp](../src/distributed/bootstrap.cpp)、
[executor/preparation.cpp](../src/runtime/executor/preparation.cpp)。

任务：协调已接收／在途的 `Failed`、进程退出和连接关闭，保留具体失败原因并补充 rank、
启动阶段上下文；避免先检查退出状态而覆盖 worker 已发送的原始错误。

完成条件：worker 模型路径不存在等实际启动错误能显示具体路径和原因；退出、断连及超时
仍能结束启动并完成所属资源清理。

## 8. KV 容量曲线的旧差分外推

- [x] 原根因已由统一 Context 池重构消除。

当前 [KV page 布局](../src/core/paged_kv_geometry.cpp)将各层、各 plane 及其对齐纳入完整
page record；容量规划直接使用真实 `page_bytes` 和统一池几何，最终资源构造仍保留相等检查。
原来的相邻页数差分公式已删除，不再安排该旧公式的修复。当时的双 Spark FP8 启动组合本轮未重跑。

## 9. Vision 前缀索引的摘要查询边界

- [ ] 分离身份查询与 checkpoint 捕获／恢复合法性。

入口：[session/request.cpp](../src/models/qwen4_exp/session/request.cpp)、
[PrefixIndex](../src/runtime/engine/context_cache/prefix_index.h)。

任务：合法输入范围内的摘要查询不受 `can_checkpoint()` 限制，使图像内部的索引分支也能
完成身份比较；捕获和恢复继续检查各自的合法边界，保持现有容量和保留策略。

完成条件：Vision 续接可找到索引分支之后的有效历史；换图不会误复用；图像内部非法
checkpoint 捕获和恢复仍被拒绝。

## 10. 双 Spark MTP grammar 的 schema 切换复用断言

- [ ] 固定前提并完成定性。

参考：[现有 grammar 用例](../tests/models/qwen3_5/test_engine_grammar_real.cpp)。

任务：固定失败序列、部署、缓存容量和保留前提，分别核对 schema 输出、历史是否仍保留、
索引候选与实际来源选择；明确该场景要求复用的合同依据，再决定是否存在引擎缺陷。
保留原始负载，不通过更换提示词或增加预热追求断言通过。

完成条件：给出有证据的定性及所属模块；确认违约后再确定修复范围。未命中本身不能直接
定性为约束解码或 MTP 数值错误。

## 验证范围

优先复用现有用例和真实产品入口，新增测试只覆盖具体风险。上述任务不包含八组测试的
机械迁移、重复测试入口、测试框架改造或新的性能对比活动。GLM5Next 集成与 Qwen3.5 的
Spark 移植独立推进；稳定文档在阶段性完成后集中整理。
