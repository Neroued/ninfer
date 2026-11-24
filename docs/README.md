# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [Startup configuration](startup-configuration.md) | shared CLI/serve TOML schema, defaults, overrides, resource controls and startup display |
| [Deployment configurations](../examples/configs/) | RTX 5090, single Spark and dual-Spark TOML examples |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

Published artifact downloads are listed in the [project README](../README.md).
[Versioned model cards](../model-cards/) own each artifact's identity, provenance and evaluation results.

## Internal architecture

内部文档记录跨模块设计、重要数值取舍和开发方法。具体类型、接口、支持几何与 codec
以所属代码为准，不在文档维护重复声明。

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | 所有权、启动预算、请求与执行事务、提交和退役 |
| [Distributed execution](maintainer/distributed-execution.md) | 集中调度、rank 准备、控制与数据通信、完整完成与失败 |
| [Context management](maintainer/context-management.md) | KV/State 统一容量、身份、预留、前缀策略、ADP 与恢复 |
| [Frontend](maintainer/frontend.md) | 模型策略与共享文本/媒体基础、输入身份、输出与 grammar 事务 |
| [Weights and loading](maintainer/weights-and-loading.md) | 表示、绑定、转换、保值分片物化与 backing |
| [Qwen3.5](maintainer/qwen3_5-model.md) | Dense/MoE、Main/MTP/DFlash、Vision 与状态组合 |
| [Qwen4Exp](maintainer/qwen4_exp-model.md) | Flash Next、HC/QSA/PLE、量化、TP/EP 与执行组合 |
| [GLM5Next](maintainer/glm5next-model.md) | KDA/MLA/mHC、量化、ADP、MTP/Vision 与 Graph |
| [Speculative execution](maintainer/speculative-decoding.md) | MTP 对齐、ReplaySSM、DFlash 分布与接受前缀 |
| [Expert cache](maintainer/expert-weight-cache.md) | 固定与流式缓存、CPU/GPU 分工、上传和消费寿命 |
| [Observability](maintainer/observability.md) | 时间和资源口径、测量窗口、rank 统计与日志边界 |
| [Op development](maintainer/op-development.md) | 语义、实现组织、数值验证与调优方法 |
| [Build system](maintainer/build-system.md) | Target、编译与链接、源码和依赖归属 |

## Development tools

- [Tests](../tests/README.md)：测试组织、筛选与真实模型检查。
- [Benchmarks](../bench/README.md)：Engine/Op 微测、roofline 与外部 profiling。
- [Tools](../tools/README.md)：转换检查、Serve runners 与带宽工具入口。
- [Capability evaluation](../eval/README.md)：API 评测、并发与结果恢复。

## Open work

[未完成事项](issue-checklist.md)保留待处理问题与 MLA 性能工作。
Qwen4-27B 的[架构与量化](designs/qwen4-27b-rtx5090.md)及
[Op 需求](designs/qwen4-27b-op-checklist.md)属于独立适配计划。

Qwen3.8-Flash-Next 使用 Qwen4Exp；Qwen3.8-27B 使用 Qwen3.5。
