# Serve TTFT 与调度测量

Runner 向已运行服务发送公开 HTTP 请求，记录首输出、后续流事件、结束与取消。
Campaign controller 另行负责启动配置与逐 sample 的服务生命周期；二者不调用模型内部接口。

## Measurement contract

请求 JSON 和媒体编码在计时前准备，每条请求使用已连接的 HTTP 连接。
TTFT 从写请求前开始，到第一个非空 text、reasoning 或 tool-argument delta 结束。
响应头、角色声明和空增量不算首输出；首输出前失败没有 TTFT，之后失败保留已测值。

TTFT 包含上传、准备、排队与首次计算。后续报告 stream gaps、完成、取消和 transport closure。
SSE event 不等于 token：spec 一次可发布多 token，网络也可能合并事件。

四种结果分别解释：请求图是否按约定构造、请求 outcome、所需机制是否实际观察到、性能数据。
`not_observed` 与 `unavailable` 不同，慢或失败的已构造场景仍是有效证据。
吞吐使用实际成功输出和完整 makespan，不用缺失 usage 的 bytes/event 数推算 token。

## Workloads and resources

[Cases](cases.py)定义请求图，[profiles](profiles.py)定义服务配置，
[冻结语料](../../../bench/fixtures/ttft/README.md)定义输入身份与 token 事实。
内置 profiles 针对 Qwen3.8-27B NVFP4/FP8；换模型需核对其 tokenizer/template 和真实资源布局。

场景包括冷热续接、共享前缀、工作集变化、Host 轮换、抢占/Replay、Vision、agent 历史和混合到达。
具体集合由代码维护，名称不保证旧策略下的某种受害者仍会出现。
资源压力按共享 Context 的真实分配、State、history 与引用判断，不恢复旧的独立槽数假设。

固定到达记录计划时间与实际发送延迟，输出遵循 EOS；基于生成历史或压力事件的负载标记为
条件性场景。改变请求图或测量合同后不能直接并入旧样本。

## Run a campaign

```bash
python3 tools/bench/run_serve_ttft_campaign.py \
  --serve build/rtx5090-dev/apps/ninfer-serve \
  --artifact /explicit/qwen3_8_27b_nvfp4.ninfer \
  --campaign resource --samples 1 --output-dir .agent-tmp/ttft/run
```

可通过重复 `--case` 选择场景，完整 catalog 和参数见 `--help`。
`--profile-config` 使用 JSON 的 `common_args` 与 `profiles` 替换内置参数表，manifest 记录实际命令。
比较不同版本时显式匹配物理预算、artifact、backend、观测和硬件条件。

每个 sample 使用新进程，warmup 在测量前完成。输出包括 manifest、raw client JSON、progress、
Serve 日志、request JSONL 和 summary；失败也保留已有测量。
Raw 输入和输出供后续身份分析，媒体 data URL 在测量后去重保存。

访问独立管理的服务：

```bash
python3 tools/bench/run_serve_ttft.py \
  --base-url http://127.0.0.1:18080 --case shared-fanout \
  --profile-label shared-prefix \
  --request-log-jsonl .agent-tmp/ttft/server.jsonl \
  --output .agent-tmp/ttft/shared-fanout.json
```

Profile label 是记录标签，不验证该服务的实际设置。

## Interpretation and comparisons

```bash
python3 tools/bench/summarize_serve_ttft.py .agent-tmp/ttft/candidate \
  --baseline .agent-tmp/ttft/baseline
```

Client 测量与 server 内部观测通过真实 wire IDs 关联，不跨进程相减绝对时钟。
Server first-output snapshot 可解释准备、queue、binding、pause、resident 和 HTTP residual；
细分 GPU/Host/传输时间可能重叠，不能再次加到 wall time。缺失数据保持 unavailable，
负 residual 保留为不一致测量。

启动资源保留 ranks 与 total allocations，独立于是否取得运行期 interval。
Periodic occupancy 是采样最大值，只有 allocator 的真实高水位称为 peak。
未完成目的仍占预留，自己的 Replay 不计跨请求 cache hit。

分析 miss 先核对真实输入与完整 checkpoint 是否存在，再看选择与回收。
最长相同 token 前缀不证明那个位置有 recurrent State；语义可复用上界与实际可保留恢复点分开解释。

比较保留每次样本、离散性、失败与未构造场景。单 sample 只说明轨迹，
显著异常才继续定位；微小差值不独立证明回退。精确 State/KV、lease 和事务边界由
[行为测试](../../../tests/README.md)验证，HTTP 测量补充真实调度与时延证据。
