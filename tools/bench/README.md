# Serve 测量工具

这里提供固定语料的服务测量与外部 TTFT 客户端。细粒度 Engine chunk/step 和 Op 成本使用
[ninfer_bench 与 Op benches](../../bench/README.md)。这些工具各自保留真实请求和计时边界，
不从另一层的吞吐推导本层性能。

## Serving corpus benchmark

[run_serve_corpus.py](run_serve_corpus.py)按固定语料串行请求，显式选择 `LABEL=PATH` artifact、
Serve executable、backend/mode、sampling 和 KV 表示。它启动服务并保存请求结果、摘要与 server
日志。准确参数、默认值和结果版本由脚本维护。

```bash
python3 tools/bench/run_serve_corpus.py --help
```

报告直接消费生产日志的 server TTFT。Vision stream 时间结合样本数解释，未测量不能写成零。
Host exposure 是请求时延分量，不跨并发直接累加为 worker 工作量。
公开比较的语料与汇总规则见[服务测量方法](../../docs/performance/methodology.md)。

## Concurrent serving benchmark

[run_serve_concurrency.py](run_serve_concurrency.py)提供 `decode-saturation` 与 `corpus-makespan`。
每个并发点建立对应服务实例；并发、context、chunk、KV 和 proposal head 明确记录。
资源报告保留各 rank 与总量，缺少启动观测明确标记 unavailable。

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/rtx5090-dev/apps/ninfer-serve \
  --artifact model=/explicit/model.ninfer --kv-dtype fp8 \
  --mode mtp3 --suite corpus-makespan --concurrency 1 --concurrency 4 \
  --output .agent-tmp/serve-bench/corpus
```

两种 suite 的工作量和时间边界不同，报告使用实际完成输出及请求 outcome。
语料会影响 expert 激活与 speculative 接受情况；重复文本不能未经测量就声称不影响结果。

## External Serve TTFT

[TTFT 说明](ttft/README.md)定义公开 HTTP 的首输出、stream gap、取消与缓存诊断。
Runner 使用冻结语料，campaign controller 负责配置服务和逐 sample 的隔离：

```bash
python3 tools/bench/run_serve_ttft_campaign.py \
  --serve build/rtx5090-dev/apps/ninfer-serve \
  --artifact /explicit/model.ninfer --campaign resource --samples 1
```

单 sample 说明该次轨迹，不建立分布结论。机制未观测到、数据不可用、请求失败与时延较高分别报告。

## Token corpus 工具

[make_bench_corpus.py](make_bench_corpus.py)为需要 token-ID fixture 的测试或实验离线生成语料，
使用显式本地 tokenizer。它不再是 `ninfer_bench` 的输入接口；后者使用 prompt/messages。

```bash
python3 tools/bench/make_bench_corpus.py --check
```

重建时按 `--help` 指定 tokenizer、源文本与输出路径。临时数据、请求结果、trace 和性能报告
统一放在 `.agent-tmp/<task>/`。
