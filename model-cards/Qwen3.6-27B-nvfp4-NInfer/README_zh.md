---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-27B
  - rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
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
            value: 84.34
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# NInfer 版 Qwen3.6-27B NVFP4

[English](README.md) | 简体中文

本模型卡是 [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer)
的版本控制源文件。

本仓库包含 [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) 的 NVFP4 表示，
由 [rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
中固定的打包权重转换为 [NInfer](https://github.com/Neroued/ninfer) 原生 `.ninfer` 产物格式。
该产物仅供 NInfer 使用，不是 Transformers checkpoint、Safetensors 分发包或 GGUF 文件。

该产物使用 Qwen3.5 Dense 架构与混合 NVFP4/BF16 权重。实际绑定与激活权限决定所选的原生执行路径，
包括 W4A4 Tensor Core prefill 与 A16 NVFP4 解码。Text、Vision、MTP、前缀复用、CLI 和服务
均走公共 Engine 路径。

## 产物

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_6_27b_nvfp4.ninfer` |
| 大小 | 18,324,354,820 bytes (17.07 GiB) |
| SHA-256 | `0448262d15df2ae4fda761540c110bc19e7c3b4f43c0938e4d50474429cda083` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5ForCausalLM` |
| 公开模型名 | `qwen3.6-27b` |
| 对话模板 | [qwen3_6.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_6.jinja); 可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 启用 thinking；省略已结束轮次的推理内容 |
| 存储对象数 | 1,545（1,539 个张量和 6 个资源） |
| NVFP4 张量 | 247 |

文件内含 Text、Vision、MTP、优化后的 proposal head 和前端资源。文本线性层沿用源仓库的
混合 NVFP4/BF16 分配；词表与 MTP 投影使用 Q8。Vision 与投机解码权重仅在启动时选用后才加载。

用以下命令校验下载文件：

```bash
printf '%s  %s\n' \
  '0448262d15df2ae4fda761540c110bc19e7c3b4f43c0938e4d50474429cda083' \
  'qwen3_6_27b_nvfp4.ninfer' | sha256sum --check
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
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

./build/rtx5090-release/apps/ninfer models/qwen3_6_27b_nvfp4.ninfer \
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
./build/rtx5090-release/apps/ninfer-serve models/qwen3_6_27b_nvfp4.ninfer \
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

以下单请求服务测量在 NVIDIA GeForce RTX 5090 上采集，使用 CUDA 13.1 编译/运行时与
CUDA driver API 13.3。请求以串行方式提交给常驻的 `ninfer-serve` 进程，启用 CUDA Graph、
1,024-token prefill 分块、INT8 group-64 KV 缓存，并关闭前缀复用。每个单请求数值为
5 个固定种子下的算术平均 ± 样本标准差；服务预热在正式测量前完成。

### 并发 MTP=3 解码饱和测试

并发测试为每个活跃请求先输入一个 293-token prompt，再生成 8,192 个 token。
每个并发度点都启动全新的服务进程，配置 MTP3、INT8 group-64 KV、CUDA Graphs、
16,384-token 单请求上下文上限，并关闭前缀复用。聚合吞吐只统计实际解码批大小保持等于 C 的
完整 1 秒区间。每行对应一轮持续波次。

| C | 稳态解码 (tok/s) | 相对 C1 加速 | 波次 makespan |
|---:|---:|---:|---:|
| 1 | 202.4 | 1.00× | 40.46 s |
| 2 | 399.7 | 1.97× | 41.82 s |
| 4 | 699.7 | 3.46× | 47.92 s |
| 8 | 1,146.9 | 5.67× | 58.57 s |

在 C=8 时，该配置维持 **1,146.9 聚合解码 tok/s**，相当于 C=1 吞吐的 **5.67×**。

### 长上下文基线（关闭 MTP）

| Prompt tokens | Prefill 阶段 (tok/s) | 服务端 TTFT (ms) | 解码阶段 (tok/s) |
|---:|---:|---:|---:|
| 7,680 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

在 7,680 prompt tokens 下，这是已发布 groupwise-int 配置 prefill 吞吐的 3.48×；
在 260,096 prompt tokens 下为 1.55×。

### MTP=3 长推理解码

启用 thinking，输出上限为 65,536 tokens。

| AIME 2026 fixture | 完成 token 数 | 解码阶段 (tok/s) | MTP 接受率 | MTP tokens/round |
|---|---:|---:|---:|---:|
| 题目 1 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| 题目 15 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| 题目 30 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

### MTP=3 跨场景解码

每个类别包含 3 个 fixture、每个 fixture 5 个种子（15 个样本）。关闭 thinking，
输出上限为 4,096 tokens。

| 类别 | 解码阶段 (tok/s) | MTP 接受率 | MTP tokens/round |
|---|---:|---:|---:|
| 代码 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| 故事 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| 翻译 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| 结构化输出 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

完整的方法与结果（含指标定义、对比数据和精确的复现命令）见
[完整方法与结果](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.6-27b.md)。

## 评测

当时的服务版本 revision 为 `b3d4d0f50b868711c62432bbd68e746217a2f49a`。
当前评测命令记录在
[评测指南](https://github.com/Neroued/ninfer/blob/master/eval/README.md) 中。
以下结果保留其原始配置。

该产物通过 NInfer 兼容 OpenAI 的服务路径评测，启用 thinking、MTP=3，
上下文上限 262,144 tokens。EvalScope 1.9.0 使用 0-shot prompt、规则评分、每题一次采样，
参数为 temperature 0.6、top-p 0.95、top-k 20、presence penalty 1.0、seed 42。
全部 258 个配置样本均完成并计分。

| 基准 | 准确率 | 正确数 / 总数 |
|---|---:|---:|
| AIME 2025 | 93.33% | 28 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 84.34% | 167 / 198 |

这些是所述 NInfer 评测配置下的单样本结果，不是 pass@k 成绩。

## 限制

- 该产物面向单张 RTX 5090，每个 Engine 支持 1 至 8 个活跃请求。
- KV 与续写状态共享启动时创建的 Context 池。可用上下文取决于所选 KV 类型、
  启用的组件、并发度和可用内存。
- NInfer 将生成的工具调用返回给客户端，不负责执行。

## 溯源

| 字段 | 值 |
|---|---|
| 基础仓库 | `Qwen/Qwen3.6-27B` |
| 基础 revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| NVFP4 源仓库 | `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| NVFP4 源 revision | `9b5389d4a1e207daab2d47732efea57d7e946dcf` |
| 转换配方 | `qwen3_6_27b_nvfp4` |
| 转换器仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时 revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |
| Ranking 输入 SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

产物标识、对象清单摘要与转换溯源发布于
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer/blob/main/artifact-manifest.json)。
精确的存储契约维护在
[v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/weights-and-loading.md) 中。

## 许可证

本 NInfer 产物按 Apache License 2.0 分发。
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) 基础仓库和
[NVFP4 源仓库](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
同样采用 Apache-2.0。用户仍需自行遵守许可证及适用法律法规。
