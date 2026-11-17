# NInfer

[English](README.md) | 简体中文

> 面向 RTX 5090 与 DGX Spark 上特定模型的原生 C++/CUDA 推理。

NInfer 通过本地 CLI 或兼容 OpenAI / Anthropic 的 HTTP API 运行文本与图像/视频生成。它支持单设备执行，以及跨两台 DGX Spark 的模型并行执行，并具备 CUDA Graph 解码、投机解码和共享前缀缓存。一个 Engine 只服务一个模型，最多支持 8 个并发活跃请求。

## 支持的模型与部署

| 模型 | 架构 | 部署方式 | 投机解码后端 |
|---|---|---|---|
| Qwen3.6-27B / Qwen3.8-27B | Qwen3.5 Dense | RTX 5090 | MTP；Qwen3.8-27B 可搭配配套权重使用 DFlash2 |
| Qwen3.6-35B-A3B | Qwen3.5 MoE | RTX 5090，常驻或 CPU 专家卸载 | MTP 或搭配配套权重的 DFlash |
| Qwen3.8-Flash-Next | Qwen4Exp | 带 CPU 专家卸载的 RTX 5090；单台或双台 Spark | MTP |
| GLM-5.3-Flash | GLM5Next | 双台 Spark | MTP |

以上所有组合均支持文本、视觉、工具调用、结构化输出和离线困惑度评分。视觉与投机解码要求产物中包含相应组件，且必须在启动时启用。评分模式仅加载文本组件。CPU 专家卸载仅适用于 RTX 5090 目标；在 Spark 上执行时，选定权重保持在 GPU 上。Flash Next 的卸载还需要大量系统内存来存放专家权重，参见 [CLI 指南](docs/cli.md#cpu-expert-offload)。

双 Spark 的 Flash Next 使用 TP2，可选叠加 EP2；GLM 使用 TP2 加注意力 DP2，可选叠加 EP2。[部署指南](docs/startup-configuration.md#distributed-deployment) 说明了这些设置与 worker 启动方式。[示例配置](examples/configs/) 覆盖了常见部署，包括视觉、FP8 KV、MTP3 和优化后的 draft head。

## 构建

在 Linux 上原生构建：RTX 5090 对应 x86-64 / `sm_120a`，DGX Spark 对应 AArch64 / `sm_121a`。依赖包括支持目标架构的 CUDA Toolkit、CMake 3.28+、C++20 主机编译器、Ninja、`pkg-config`、libjpeg 开发头文件、FFmpeg 开发库（`libavformat`、`libavcodec`、`libavutil`、`libswscale`）、`hwloc >= 2.5`、`liburing >= 2.5` 以及 `libcurl >= 7.85`。开发使用 CUDA 13.4；CMake 会检查所请求的架构。

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release -j
```

单台 Spark 请使用 `dgx-spark-release` 预设和 `build/dgx-spark-release` 目录。双 Spark 需安装 NCCL 开发头文件与库，并**在两个节点上都**完成构建：

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

产品二进制为 `apps/ninfer`、`apps/ninfer-serve`、`apps/ninfer-perplexity`，启用分布式执行时还包括 `apps/ninfer-worker`。请在构建树内运行它们；移动产物时，需将构建树中的 `lib/` 目录与 `apps/` 保持同级。本项目不提供可安装的 C++ SDK。

产品预设不包含测试和基准。`rtx5090-dev` / `dgx-spark-dev` 预设会同时启用二者；对应的 `-ops` 预设只构建底层库及其检查。开发选项参见[构建组织](docs/maintainer/build-system.md)。

## 获取权重

Engine 读取 v3 版 `.ninfer` 产物，其中包含权重、配置、分词器、对话模板和组件资源。官方下载如下：

| 模型 | 权重 | 产物 | 下载与模型卡 |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8-Flash-Next | 混合 NVFP4/Q8/NF4 | `qwen3_8_flash_next.ninfer` + 3 个分片 | [Qwen3.8-Flash-Next](https://huggingface.co/neroued/Qwen3.8-Flash-Next-NInfer) |
| GLM-5.3-Flash | 混合 NVFP4/Q8 | `glm5_3_flash.ninfer` + 5 个分片 | [GLM-5.3-Flash](https://huggingface.co/neroued/GLM-5.3-Flash-NInfer) |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models
```

下载 Flash Next 或 GLM 时需一并获取全部分片：

```bash
hf download neroued/Qwen3.8-Flash-Next-NInfer \
  --include 'qwen3_8_flash_next.ninfer*' \
  --local-dir models/Qwen3.8-Flash-Next-NInfer

hf download neroued/GLM-5.3-Flash-NInfer \
  --include 'glm5_3_flash.ninfer*' \
  --local-dir models/GLM-5.3-Flash-NInfer
```

Flash Next 共 4 个文件、101.00 GiB；GLM 共 6 个文件、178.33 GiB。请把每个分片与入口文件放在同一目录，且只将入口路径传给 NInfer。双 Spark 部署时，每个节点都需放置完整的一套；加载期间会按 rank 切分。模型卡介绍了各自的混合精度、组件和部署选项。

自定义权重请参考[权重转换](docs/weight-conversion.md)。旧的 v2 文件可以[在本地升级](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)。

## 运行

在 RTX 5090 上运行单次请求：

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode briefly." \
  --max-context 8192 --max-tokens 1024 --kv-dtype fp8
```

回答内容输出到 stdout；思考过程、启动信息和结果报告输出到 stderr。图像/视频输入请使用 `--messages FILE` 和 `--vision`。参见 [CLI 指南](docs/cli.md)。

如需常驻服务，修改部署示例后启动：

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
# 修改 model.path，指向你的产物。
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

该示例启用了视觉、FP8 KV、MTP3 和优化后的 draft head。它将单请求上下文上限设为 240,000 tokens；实际可用的共享 KV/State 容量取决于可用内存。启动日志会报告 Context 池大小以及估算的 token 容量。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

服务默认只监听回环地址。如需局域网访问，请设置 `server.host = "0.0.0.0"`，并按需配置 `server.api_key`。[Serve 指南](docs/serving.md) 介绍了端点、工具调用、结构化输出、流式输出、指标和请求日志。

两个程序都接受 `--config FILE`；命令行显式传入的值会覆盖配置文件。根目录下的 [ninfer.example.toml](ninfer.example.toml) 列出了所有字段，[部署示例](examples/configs/) 则提供针对性的起点。[启动配置](docs/startup-configuration.md) 说明默认值、资源控制和分布式启动。个人 TOML 文件已被 Git 忽略。

## 上下文与前缀复用

`max_context` 限制单次请求的输入与生成 token 之和。`context_capacity` 控制另一块在启动时按字节定容、每个 rank 独立的池，由不断增长的 KV 历史和续写状态共享。设为 `auto` 时，该池使用权重与执行资源规划后剩余的可用预算。它不会随并发数倍增，逻辑上的上下文上限也不保证每个活跃请求都能同时达到该长度。

Serve 会保留精确的文本与多模态前缀，包括续写所需的状态。内存压力下可回收已缓存的前缀，或暂停活跃请求以便后续恢复/重放。RTX 5090 还可把上下文保留在锁页主机内存中；Spark 则使用设备侧上下文存储。调优参见[资源控制](docs/startup-configuration.md#context-and-memory)。

## 性能

以下记录的测试均基于 RTX 5090 和各结果页所标明的引擎版本。[性能索引](docs/performance.md) 链接到各模型的运行记录与[测量规则](docs/performance/methodology.md)。下表摘自这些详细结果。这些测试对 Qwen3.8 使用 FP8 E4M3 row-256 KV，对 Qwen3.6 使用 INT8 group-64 KV。测量数据保留其原始缓存格式与资源设置，并非当前 Spark 或分布式路径的测量结果。

### 并发 MTP3 解码

饱和解码使用 CUDA Graphs、MTP3，每个活跃请求生成 8,192 个 token。吞吐取实际解码批大小等于所配置并发度的完整区间内的聚合已提交解码 token；接受率覆盖整轮完整请求；这些速率为稳态解码（tok/s）。

| 模型配置 | C=1 tok/s / 接受率 | C=2 tok/s / 接受率 | C=4 tok/s / 接受率 | C=8 tok/s / 接受率 |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### 单请求服务

串行服务语料使用 CUDA Graphs、1,024 token 的 prefill 分块，以及预热后的 5 个固定种子。下表为每个已发布配置保留一个短 prefill、一个超长 prefill 和一个结构化输出 MTP3 数据点；完整的上下文与场景矩阵见下方各模型链接。

| 模型配置 | 7,680-token prefill | 260,096-token prefill | 结构化 MTP3 解码 |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

## 评测

下表 27B/35B 的能力分数通过 NInfer 兼容 OpenAI 的服务路径测得，启用了 thinking、MTP3，并使用 EvalScope 1.9.0（0-shot、规则评分、每题一次采样）：

| 模型配置 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 各行使用 temperature 0.6、presence penalty 1.0；Qwen3.8 各行使用 temperature 1.0、presence penalty 0.0。多模态评测使用 `--vision` 和 81,920 token 的上下文上限。文本评测使用 262,144 tokens，但 Qwen3.8-27B NVFP4 使用 252,928 tokens，以便在加载权重后适配 RTX 5090。每个分数均为每题一次采样；模型卡中给出正确数/总数及评测说明。

[Flash Next](model-cards/Qwen3.8-Flash-Next-NInfer/README.md#evaluation) 单独在双 Spark 上以 TP2/EP2、BF16 KV、视觉、MTP3、优化后的 draft head 和 EvalScope 1.10.0 进行评测：

| 基准 | 准确率 | 正确数 / 计分样本数 | 每题重复次数 |
|---|---:|---:|---:|
| GPQA-Diamond | 91.79% | 727 / 792 | 4 |
| AIME 2026 | 97.78% | 88 / 90 | 3 |
| ERQA | 68.00% | 272 / 400 | 1 |
| RealWorldQA | 85.75% | 656 / 765 | 1 |
| MMMU-Pro，标准 10 选项 | 80.81% | 1,398 / 1,730 | 1 |

这些是样本平均准确率，而非 pass@k。模型卡记录了生成设置，并把截断/未解析的样本计入总数。视觉分数保留原始测试中的解码器与评分行为。

## Docker

随附的 Dockerfile 构建 RTX 5090 的 CLI 与服务端。宿主机需安装 NVIDIA Container Toolkit：

```bash
docker build --tag ninfer:local .
docker run --rm --gpus '"device=0"' --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --max-context 8192 --kv-dtype fp8
```

## 文档

- [部署配置](examples/configs/) 与 [启动参数](docs/startup-configuration.md)
- [CLI](docs/cli.md)、[HTTP 服务](docs/serving.md) 和 [困惑度](docs/perplexity.md)
- [权重转换](docs/weight-conversion.md)
- [性能记录](docs/performance.md)、[基准](bench/README.md) 和 [评测](eval/README.md)
- [文档索引](docs/README.md) 与 [贡献指南](CONTRIBUTING.md)

## 许可证

NInfer 采用 [Apache License 2.0](LICENSE)。

27B 与 35B-A3B 的产物源使用 Apache-2.0，具体记录见各自模型卡。Flash Next 结合了采用 [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) 的源码与采用 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 的权重。GLM 的源权重采用 [MIT](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)。模型许可证与 NInfer 的软件许可证相互独立。`third_party/` 下的内置依赖保留各自的许可证文件。
