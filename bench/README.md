# Benchmarks

选择与问题相符的层级：`ninfer_bench` 测公共 Engine 的执行与资源，Op benchmark 测完整数学入口，
[Serve runners](../tools/bench/README.md)测固定语料和 HTTP 并发，
[TTFT](../tools/bench/ttft/README.md)测请求到达、缓存和流式延迟。
数值正确性由 tests 验证，调优方法见 [Op 开发](../docs/maintainer/op-development.md#调优与测量)。

## 构建与入口

使用包含 benchmarks 的开发 preset：

```bash
cmake --preset rtx5090-dev
cmake --build build/rtx5090-dev -j
./build/rtx5090-dev/bench/ninfer_bench --help
```

Spark 使用 `dgx-spark-dev`；分布式配置需启用分布式构建。`*-ops` 只提供对应基础和 Op benches。
参数和支持点由各 executable 的 `--help` 与源文件维护。临时输入、CSV、报告和 profiler 产物
统一写入 `.agent-tmp/<task>/`。

## Engine 微测

`ninfer_bench` 使用公共启动配置和正常请求，不提供私有模型循环或状态恢复接口。
用 positional artifact 或 `--config` 选择模型，prompt/messages 通过命令行提供：

```bash
./build/rtx5090-dev/bench/ninfer_bench --config ninfer.local.toml \
  --messages .agent-tmp/bench/input.json --max-tokens 128 \
  --concurrency 4 --warmup 1 --repetitions 3 \
  --output json --output-file .agent-tmp/bench/result.json
```

`--prompt` 与 `--messages` 可以重复，提供一个输入时复制为 C 条，提供多个时数量须与并发相同。
消息文件使用 CLI 的 conversation 格式。TOML 提供引擎与生成配置，不读取其 `[cli]` 输入作为语料。
KV、Vision、backend、draft、proposal head、Graph 与 deployment 沿用公共启动参数。

`--max-tokens 1` 用于 Prefill；更大值测正常生成。输出上限不是 decode step 数，遵循 EOS。
每轮从相同输入和 sampling seeds 开始正常请求，保留运行时缓存预热。Decode 测量前先让前 C−1 个
输入完成普通单 token 请求，再一起提交正式请求，以利用已有前缀缓存。它不强制调度达到 C，
也不保证所有 rows 同时存活。Prefill-only 默认不复用前缀，显式配置优先。

报告重点为 Prefill chunk 与实际 batch 下的 decode step：样本数、mean/max、Result、Resolution、
Retirement；没有目标 batch 样本就明确为空。并列给出原生 GPU/Host 观测、spec 工作和资源，
不从平均 batch 推算目标并发性能。各次 repetition 和汇总保留在 JSON 中。
计时边界见[观测文档](../docs/maintainer/observability.md)。

### 外部 profiling

`--profile-measured` 要求 `--repetitions 1`，通过 Engine 在实际执行 rank 开启/关闭 profiler。
准备、预热与 C−1 个请求在窗口外，结束等待最终异步退役。单机可使用：

```bash
nsys profile --capture-range=cudaProfilerApi \
  --output=.agent-tmp/bench/decode \
  ./build/rtx5090-dev/bench/ninfer_bench --config ninfer.local.toml \
  --messages .agent-tmp/bench/input.json --max-tokens 16 \
  --warmup 1 --repetitions 1 --profile-measured
```

双机分别在 worker 进程采集；协调进程上的 nsys 不会自动记录远端 CUDA。
API 测量边界由 Engine 统一下发，采集进程与报告文件仍由外部 profiler 启动方式决定。

## Op 测量

Op benchmark 调用公开入口及其 workspace 查询，准备、初始化和状态恢复放在计时之外。
输入按实际表示产生，控制量和 expert 分布保持所声明的意义。原位 Op 每次测量从适当初态开始，
不能在同一 Graph 中反复改变状态后把结果当作独立重复。

Cold 模式在计时前读 buffer 驱逐 L2，warm 模式单独报告。Graph launch 计时与 graph body 事件计时
有不同边界；一次 graph 内的多调用可以复用缓存，不能称每次都是冷读。
比较前固定 cache、timer、调用数、warmup/repeats 和实际 GPU。

Linear 示例：

```bash
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 6144 --k 5120 --sweep 1:128:1 \
  --execution graph --graph-timer body --cache cold \
  --warmup 3 --repeat 20 --csv-out .agent-tmp/linear/latency.csv
```

一次精确点可以使用 `--t`；profile 只采必要的公开调用。长期工具不强制私有 kernel。
GDN 的 stage/crossover 诊断保留其明确例外，prepare、recurrence 或预归一化输入的结果
不能代替完整 Op。其他 family 的语义、模式和默认样本直接查所属入口：

| 问题 | 代码入口 |
|---|---|
| Linear、split、融合投影 | [Linear benches](ops/)及各 executable `--help` |
| GDN/KDA update、record、fold 与 chunked | [Linear attention](ops/linear_attention/) |
| GQA/QSA/Kpool 的 prepare、indexer、attention 与组合链 | [Softmax attention](ops/softmax_attention/) |
| 路由分布、expert TP/EP、完整 MoE | [MoE](ops/moe/) |
| 模型组件与资源成本 | [Model benches](models/)、[Context cost](context_cost/) |

## Hardware references

[roofline.h](common/roofline.h)维护编译目标的硬件参照，新增来源或校准只在该处及必要依据中记录。
有效 GB/s、useful TFLOP/s 使用逻辑工作量，不等于 DRAM counters 或真实执行指令数。
Tensor Core 参照必须匹配实际生产路线的输入与 accumulator；SIMT 或未知路径不填 TC 利用率。

权重统计包含消费的 codes/scales，MoE 按实际去重 expert 集合计流量，recurrent 按 State
读写计费，chunked 还需考虑必要 workspace。Cold/warm 与重复读取影响解释，不能仅用低比特
存储压缩比宣称加速。`max(bytes/bandwidth, FLOPs/compute)` 是给定口径的理想下界，
不是已证明可达的调度。

报告保留真实逐点曲线、关键区间、实际条件与未解决异常。没有请求保留性能报告时，
不要求为每次改动额外创建图表文档。既有公开数据见[性能页](../docs/performance.md)。
