---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-27B-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 86.67
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 93.33
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 86.87
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# NInfer 版 Qwen3.6-27B

[English](README.md) | 简体中文

本模型卡是 [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer)
的版本控制源文件。

本仓库包含转换为 [NInfer](https://github.com/Neroued/ninfer) 原生 `.ninfer` 产物格式的
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)。该产物仅供 NInfer 使用，
不是 Transformers checkpoint、Safetensors 分发包或 GGUF 文件。

## 产物

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_6_27b.ninfer` |
| 大小 | 17,495,538,688 bytes (16.29 GiB) |
| SHA-256 | `9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5ForCausalLM` |
| 公开模型名 | `qwen3.6-27b` |
| 对话模板 | [qwen3_6.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_6.jinja); 可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 启用 thinking；省略已结束轮次的推理内容 |

文件内含 Text、Vision、MTP、优化后的 proposal head 和前端资源。文本投影使用 Q4/Q5，
词表权重使用 Q6，MTP 投影使用 Q8。Vision 与投机解码权重仅在启动时选用后才加载。

用以下命令校验下载文件：

```bash
printf '%s  %s\n' \
  '9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b' \
  'qwen3_6_27b.ninfer' | sha256sum --check
```

## 环境要求

- 当前 [NInfer](https://github.com/Neroued/ninfer) 检出版本，从源码构建；
- 64 位 Linux；
- NVIDIA GeForce RTX 5090（`sm_120a`）；
- [构建指南](https://github.com/Neroued/ninfer#build) 中列出的构建依赖。

已有官方 v2 文件？可[在本地升级](https://github.com/Neroued/ninfer/blob/master/docs/weight-conversion.md#upgrade-an-existing-v2-artifact)，
无需重新下载权重。

NInfer 不提供安装目标或打包好的二进制文件。源码构建依赖见
[仓库 README](https://github.com/Neroued/ninfer#build)。

## 下载并运行 CLI 示例

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

./build/rtx5090-release/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-tokens 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized
```

图像、视频与结构化对话历史参见
[CLI 指南](https://github.com/Neroued/ninfer/blob/master/docs/cli.md)。

## 启动本地服务

```bash
./build/rtx5090-release/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --context-capacity auto \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized \
  --preserve-thinking
```

每个请求的逻辑上限为 240,000 tokens。KV 历史与续写状态共享启动时选定的
Device Context 字节容量。日志会报告池大小与估算的 token 容量。请求在推进过程中按需分配；
资源紧张时可回收已保留的前缀，或暂停请求以便后续恢复/重放。容量、Host 缓存与部署控制参见
[启动配置参考](https://github.com/Neroued/ninfer/blob/master/docs/startup-configuration.md)。

API 接口见 [HTTP 服务指南](https://github.com/Neroued/ninfer/blob/master/docs/serving.md)，
缓存与准入语义见
[资源调度参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/context-management.md)。

## 支持范围

该产物支持：

- thinking 与非 thinking 模式下的文本生成；
- 图像、多图、视频及混合多模态消息；
- MTP 投机解码，草稿窗口 1 至 5；
- BF16、INT8、FP8、NVFP4 和 K8V4 KV 缓存；
- CUDA Graph 解码与兼容前缀复用；
- 启动时定容的小规模并发服务，支持真正的批量解码；
- NInfer CLI；
- OpenAI Responses Core、OpenAI Chat Completions 和 Anthropic Messages 服务。

## 性能

以下单请求服务测量在 NVIDIA GeForce RTX 5090 上采集，使用 CUDA 13.1。请求以串行方式提交给
常驻的 `ninfer-serve` 进程，启用 CUDA Graph、1,024-token prefill 分块、INT8 group-64 KV 缓存，
并关闭前缀复用。每个单请求数值为 5 个固定种子下的算术平均 ± 样本标准差；预热请求不计入。
这些结果使用 `groupwise-int` 配方。

### 并发 MTP=3 解码饱和测试

并发测试使用 CUDA driver API 13.3，每个活跃请求先输入一个 293-token prompt，再生成
8,192 个 token。每个并发度点都启动全新的服务进程，配置 MTP3、INT8 group-64 KV、CUDA Graphs、
16,384-token 单请求上下文上限，并关闭前缀复用。聚合吞吐只统计实际解码批大小保持等于 C 的
完整 1 秒区间。每行对应一轮持续波次。

| C | 稳态解码 (tok/s) | 相对 C1 加速 | 波次 makespan |
|---:|---:|---:|---:|
| 1 | 185.8 | 1.00× | 44.23 s |
| 2 | 247.0 | 1.33× | 66.67 s |
| 4 | 309.5 | 1.67× | 107.49 s |
| 8 | 535.0 | 2.88× | 125.20 s |

### 长上下文基线（关闭 MTP）

| Prompt tokens | Prefill 阶段 (tok/s) | 服务端 TTFT (ms) | 解码阶段 (tok/s) |
|---:|---:|---:|---:|
| 7,680 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

### MTP=3 长推理解码

启用 thinking，输出上限为 65,536 tokens。

| AIME 2026 fixture | 完成 token 数 | 解码阶段 (tok/s) | MTP 接受率 | MTP tokens/round |
|---|---:|---:|---:|---:|
| 题目 1 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| 题目 15 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| 题目 30 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

### MTP=3 跨场景解码

每个类别包含 3 个 fixture、每个 fixture 5 个种子（15 个样本）。关闭 thinking，
输出上限为 4,096 tokens。

| 类别 | 解码阶段 (tok/s) | MTP 接受率 | MTP tokens/round |
|---|---:|---:|---:|
| 代码 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| 故事 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| 翻译 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| 结构化输出 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

完整的方法与结果（含指标定义和精确的复现命令）见
[完整方法与结果](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.6-27b.md)。

## 评测

该产物通过 NInfer 兼容 OpenAI 的服务路径评测，启用 thinking、MTP=3，
上下文上限 262,144 tokens。EvalScope 1.9.0 使用 0-shot prompt、规则评分、每题一次采样，
参数为 temperature 0.6、top-p 0.95、top-k 20、presence penalty 1.0、seed 42。
全部 258 个配置样本均完成并计分。

| 基准 | 准确率 | 正确数 / 总数 |
|---|---:|---:|
| AIME 2025 | 86.67% | 26 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 86.87% | 172 / 198 |

这些是所述 NInfer 评测配置下的单样本结果，不是 pass@k 成绩。

## 限制

- 该产物面向单张 RTX 5090，每个 Engine 支持 1 至 8 个活跃请求。
- KV 与续写状态共享启动时创建的 Context 池。可用上下文取决于所选 KV 类型、
  启用的组件、并发度和可用内存。
- NInfer 将生成的工具调用返回给客户端，不负责执行。

## 溯源

| 字段 | 值 |
|---|---|
| 源仓库 | `Qwen/Qwen3.6-27B` |
| 源 revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| 转换配方 | `qwen3_6_27b` |
| 转换器仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时 revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |

产物标识、对象清单摘要与转换溯源发布于
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/blob/main/artifact-manifest.json)。
精确的存储契约维护在
[v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/weights-and-loading.md) 中。

## 许可证

本 NInfer 产物按 Apache License 2.0 分发。源
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) 仓库同样采用 Apache-2.0。
用户仍需自行遵守许可证及适用法律法规。
