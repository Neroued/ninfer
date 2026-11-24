# 观测与日志

观测描述实际发生的工作、资源和时间。生产者拥有事实，公共类型承担交接，产品层决定展示。
Pretty 日志、JSONL、metrics 与 bench 不通过解析彼此的文本获得数据。

## 所有权

Engine 记录请求生命周期、调度工作和协调侧执行边界；模型/Executor 记录真实工作与参与 rank
完成；RankProgram 记录设备阶段与资源；存储 owner 报告分配和传输。消费者直接使用共同观测，
不再根据模型名称或阶段耗时推算另一份事实。

类型入口：[执行观测](../../include/ninfer/observability/execution.h)、
[资源观测](../../include/ninfer/observability/resources.h)、
[快照](../../include/ninfer/observability/snapshot.h)。字段与序列化随这些代码维护。

## 时间的边界

| 指标 | 解释 |
|---|---|
| 请求 TTFT | 首次有效输出对应的请求边界；不以 prepare、Vision、prefill 相加代替 |
| 执行 Result | 从提交到结果可供控制侧读取 |
| Resolution | 结果可用之后的解释、决定及提交交接区间 |
| Retirement | 后续设备提交与最终资源退役区间 |
| GPU stream 时间 | 对应设备事件区间，可以与 Host 或另一 stream 重叠 |
| Host 工作时间 | 实际 CPU 工作区间，不等于整个请求或设备等待墙钟 |

分布式执行的组耗时来自完整完成边界；各 rank 的设备时间分别展示，不相加当作组延迟。
Prefill chunk 和 decode step 使用执行样本。Decode 按实际 batch 分组，不能用平均 batch
解释目标并发的延迟。Spec 的实际接受量与执行时间分开，token/s 不能代替每步计算成本。

首 token、结果返回、提交被接受和最后消费者退出不是同一个时刻。
Replay 单独计工作，不重复计公开输出；取消保留已经发生的工作成本。
CPU grammar、GPU forward 和通信可以重叠，细分时间不能无条件加到总 wall time。

计时关闭或没有样本时表示未测量，只有存在有效样本才能解释零值。
客户端网络 TTFT 与服务端 TTFT 保持各自边界，SSE event 数不当作 token 数。

## 资源的边界

显式分配、逻辑使用和物理内存快照分别报告。Workspace 中的 Vision 子区域、已注册的 Host
来源、expert ready/target 字节、Context relocation 等都是其父项的部分，不重复相加。

Context capacity 是 KV 与 State 共享的 backing。Token capacity 展示使用模型提供的参考工作集
与容量模型估计，说明在该配置下可承载的量级；不是划出的纯 KV 配额，也不是并发满长保证。
ADP 的独立 owner 容量与 TP 的共同分片约束由模型聚合，展示层不自行求和或取最小值。

启动预算分列可消耗 headroom、必须保留的 Host 容量和规划余量。最后的可用物理内存不是
规划余量，二者不可互换。Graph/通信准备的内存增长含驱动与惰性资源；Graph 释放后仍驻留的
部分已经计入新快照，不能再当作需要重建的全部分配。

多 rank 保留每侧观测与必要总量。缺失 rank、缺少资源或没有有效时间窗口须明确表示，
不以空对象或零值伪装成正常数据。

## 测量窗口

Engine 的 begin/end measurement 和峰值重置在控制 worker 上执行。边界等待已有请求、Context
事务和最后异步消费者结束，再由实际 CUDA owner 控制 profiler/计数；多 rank 等待全部确认。
请求结果返回后遗留的正常退役工作不作为非法忙状态。

准备输入、Graph 准备和预热放在正式窗口之外。结束覆盖最后提交，失败清理停止已经开始的
profiler 并保留原始错误。窗口可以复用自然预热的缓存，不隐式销毁请求之外的运行时资源。

使用方法见[bench](../../bench/README.md)；实现见
[Engine 测量边界](../../src/runtime/engine/engine_schedule.inl)与
[rank 测量](../../src/runtime/executor/measurement.h)。

## 日志呈现

产品持有 logging runtime 与 sink，底层通过 typed event/callback 报告事实，不依赖产品 logger。
产品结果、机器测量、操作日志和底层紧急诊断分别有自己的用途。正常路径不解析 pretty 文本。

启动进度携带 rank/node、attempt 与 phase 身份，耗时由生产者提供；嵌套阶段解释父阶段，
不能再次相加。转发有界，失败保留最终原因，日志回调异常不改变引擎准备结果。

日志包含必要的标识、计数和原因，不输出凭据、prompt、生成内容或媒体载荷。
预期请求拒绝、可恢复问题与实例失败按实际影响分类；不能用更高日志级别代替正确的错误归属。
请求机器终态与后续响应传输错误分开，避免一次请求产生两份相互矛盾的结算。

呈现与资源写出入口在 [product](../../src/product/)和 [serve](../../src/serve/)。
