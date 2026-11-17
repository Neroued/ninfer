---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-35B-A3B
  - z-lab/Qwen3.6-35B-A3B-DFlash
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-35B-A3B-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 90.0
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
            value: 90.0
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
            value: 85.35
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# NInfer 版 Qwen3.6-35B-A3B

[English](README.md) | 简体中文

本模型卡是
[neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer)
的版本控制源文件。

本仓库包含转换为 [NInfer](https://github.com/Neroued/ninfer) 原生 `.ninfer` 产物格式的
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。该产物仅供 NInfer 使用，
不是 Transformers checkpoint、Safetensors 分发包或 GGUF 文件。可选的 DFlash 配套权重来自
[z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)。

## 产物

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_6_35b_a3b.ninfer` |
| 大小 | 22,790,484,480 bytes (21.23 GiB) |
| SHA-256 | `3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5MoeForCausalLM` |
| 公开模型名 | `qwen3.6-35b-a3b` |
| 对话模板 | [qwen3_6.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_6.jinja); 可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 启用 thinking；省略已结束轮次的推理内容 |

文件内含 Text、Vision、MTP、DFlash、优化后的 proposal head 和前端资源。路由专家使用
Q4 gate/up 与 Q5/Q6 down 权重；共享专家与 mixer 投影使用 Q8。Vision 与投机解码权重
仅在启动时选用后才加载。

用以下命令校验下载文件：

```bash
printf '%s  %s\n' \
  '3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84' \
  'qwen3_6_35b_a3b.ninfer' | sha256sum --check
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
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models

./build/rtx5090-release/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-tokens 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized
```

DFlash 已测量的 block-8 配置使用 7 个 draft token：

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-tokens 8192 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 \
  --proposal-head optimized
```

`--draft-tokens` 对 MTP 接受 `1..5`，对 DFlash 接受 `1..15`。DFlash 取值 `7` 对应已测量的
block-length-eight 配置；`15` 使用配套权重完整的原生 16 位置块。MTP 与 DFlash 是互斥的后端选项。
DFlash 可与 `--vision` 组合用于图像或视频 prompt；它加速的是生成文本解码，而非 Vision 编码
或目标模型 prefill。

图像、视频与结构化对话历史参见
[CLI 指南](https://github.com/Neroued/ninfer/blob/master/docs/cli.md)。

## 启动本地服务

```bash
./build/rtx5090-release/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
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
- DFlash 投机解码，支持 Text 与图像/视频 Vision prompt，草稿窗口 1 至 15；
- BF16、INT8、FP8、NVFP4 和 K8V4 KV 缓存；
- CUDA Graph 解码与兼容前缀复用；
- 启动时定容的小规模并发服务，支持真正的批量解码；
- NInfer CLI；
- OpenAI Responses Core、OpenAI Chat Completions 和 Anthropic Messages 服务。

## 性能

测量于 2026-09-07，NInfer revision `487f89773f07cb18a2fb841fe0971ec9634d409b`，
通过单张 RTX 5090 上的公开 HTTP 服务路径完成。Release 构建使用 CUDA 13.1 编译/运行时与
CUDA driver API 13.3、INT8 group-64 KV、CUDA Graphs、1,024-token prefill 分块，并关闭前缀复用。
每个测量点都启动全新服务进程，且不计入启动预热。

### 长上下文基线

不启用投机解码后端；上下文上限 262,144 tokens；每个 prompt 长度使用 5 个固定种子。
数值为算术平均 ± 样本标准差。

| Prompt tokens | Prefill 阶段 (tok/s) | 服务端 TTFT (ms) | 解码阶段 (tok/s) |
| --- | --- | --- | --- |
| 7,680 | 17,705.4 ± 234.6 | 437.7 ± 6.0 | 338.3 ± 5.5 |
| 64,512 | 11,758.0 ± 122.3 | 5,510.0 ± 57.3 | 298.1 ± 2.0 |
| 130,048 | 8,317.1 ± 109.3 | 15,685.1 ± 208.0 | 260.6 ± 4.4 |
| 260,096 | 5,247.0 ± 30.1 | 49,657.3 ± 283.0 | 213.0 ± 3.3 |

### MTP3 单请求解码

3 个 draft token、优化后的 proposal head、随机采样。C=1 语料点为每个推理 fixture 提供
5 个样本、每个场景类别 15 个样本。推理场景启用 thinking，输出预算 65,536 tokens；
其他场景关闭 thinking，预算 4,096 tokens。

| Fixture | 完成 token 数 | 解码阶段 (tok/s) | 投机接受率 | 投机 tokens/round |
| --- | --- | --- | --- | --- |
| `long_decode_aime26_01` | 8,407.2 ± 2,764.1 | 750.6 ± 22.4 | 83.5% ± 3.8% | 3.50 ± 0.11 |
| `long_decode_aime26_15` | 64,860.2 ± 1,511.1 | 636.5 ± 9.2 | 72.0% ± 1.0% | 3.16 ± 0.03 |
| `long_decode_aime26_30` | 55,354.6 ± 7,132.4 | 683.3 ± 4.0 | 79.2% ± 1.1% | 3.38 ± 0.03 |

| 类别 | 解码阶段 (tok/s) | 投机接受率 | 投机 tokens/round |
| --- | --- | --- | --- |
| 代码 | 677.2 ± 25.6 | 70.5% ± 3.6% | 3.12 ± 0.11 |
| 故事 | 465.2 ± 36.3 | 37.8% ± 5.5% | 2.14 ± 0.16 |
| 翻译 | 659.0 ± 35.1 | 67.7% ± 5.8% | 3.03 ± 0.17 |
| 结构化 | 779.6 ± 44.3 | 87.5% ± 7.1% | 3.63 ± 0.21 |

### MTP3 语料 makespan

每个 C 运行同一份 75 请求语料。速率按完整 makespan 计算，包含 prefill、负载切换和排空。
单请求上下文上限为 262,144 tokens，KV 容量自动确定。

| C | 解码 token 数 | Makespan (s) | 请求数/s | 语料解码 (tok/s) | 平均批大小 | MTP 接受率 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 831,378 | 1,279.40 | 0.0586 | 649.8 | 1.00 | 72.2% |
| 2 | 827,334 | 898.13 | 0.0835 | 921.2 | 2.00 | 73.2% |
| 4 | 807,280 | 737.87 | 0.1016 | 1,094.1 | 3.45 | 71.7% |
| 8 | 861,416 | 679.61 | 0.1104 | 1,267.5 | 6.48 | 73.3% |

### DFlash K=7 单请求解码

7 个 draft token、优化后的 proposal head，语料与上下文上限相同。以下数值使用随机采样
和按请求统计的解码阶段耗时。

| Fixture | 完成 token 数 | 解码阶段 (tok/s) | 投机接受率 | 投机 tokens/round |
| --- | --- | --- | --- | --- |
| `long_decode_aime26_01` | 8,536.2 ± 3,506.4 | 866.8 ± 47.4 | 66.4% ± 4.0% | 5.65 ± 0.28 |
| `long_decode_aime26_15` | 65,325.0 ± 471.8 | 641.6 ± 62.4 | 49.7% ± 6.2% | 4.48 ± 0.43 |
| `long_decode_aime26_30` | 53,756.4 ± 5,693.8 | 732.6 ± 13.3 | 58.1% ± 1.4% | 5.07 ± 0.10 |

| 类别 | 解码阶段 (tok/s) | 投机接受率 | 投机 tokens/round |
| --- | --- | --- | --- |
| 代码 | 620.6 ± 43.3 | 42.6% ± 4.1% | 3.98 ± 0.28 |
| 故事 | 291.6 ± 58.8 | 12.2% ± 5.4% | 1.85 ± 0.38 |
| 翻译 | 547.5 ± 74.5 | 35.2% ± 6.8% | 3.47 ± 0.48 |
| 结构化 | 906.4 ± 127.3 | 69.7% ± 12.5% | 5.88 ± 0.88 |

### MTP3 解码饱和测试

每个活跃请求生成 8,192 个 token，上下文上限 16,384 tokens，每个 C 一轮波次。
稳态速率只统计解码批满的完整区间；接受率覆盖整轮波次。

| C | 稳态解码 (tok/s) | MTP 接受率（整轮波次） | 波次 makespan (s) |
| --- | --- | --- | --- |
| 1 | 642.5 | 68.6% | 12.70 |
| 2 | 907.2 | 66.3% | 18.03 |
| 4 | 1,213.5 | 69.6% | 27.27 |
| 8 | 1,380.7 | 68.0% | 47.94 |

11 个测量点上的全部 485 个正式请求均完成，未出现请求、CUDA 或内存不足失败。
在 225 条 C=1 投机解码输出中未发现明显的短周期重复。达到输出长度上限的样本仍保留在报告结果中。

DFlash 贪心结果、语料 token 总量、终止计数、资源设置与命令见
[完整性能报告](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.6-35b-a3b.md)。

## 评测

该产物通过 NInfer 兼容 OpenAI 的服务路径评测，启用 thinking、MTP=3，
上下文上限 262,144 tokens。EvalScope 1.9.0 使用 0-shot prompt、规则评分、每题一次采样，
参数为 temperature 0.6、top-p 0.95、top-k 20、presence penalty 1.0、seed 42。
全部配置样本均完成并计分。

| 基准 | 准确率 | 正确数 / 总数 |
|---|---:|---:|
| AIME 2025 | 90.00% | 27 / 30 |
| AIME 2026 | 90.00% | 27 / 30 |
| GPQA-Diamond | 85.35% | 169 / 198 |

这些是所述 NInfer 评测配置下的单样本结果，不是 pass@k 成绩。

## 限制

- 该产物面向单张 RTX 5090，每个 Engine 支持 1 至 8 个活跃请求。
- 当 GPU 预算无法容纳常驻配置时，主专家支持 CPU/GPU 卸载。
- KV 与续写状态共享启动时创建的 Context 池。可用上下文取决于所选 KV 类型、
  启用的组件、并发度和可用内存。
- NInfer 将生成的工具调用返回给客户端，不负责执行。

## 溯源

| 字段 | 值 |
|---|---|
| 基础源仓库 | `Qwen/Qwen3.6-35B-A3B` |
| 基础源 revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| DFlash 源仓库 | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| DFlash 源 revision | [`f181eece646affea2c38b2765f1aaa01a9734ccd`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/tree/f181eece646affea2c38b2765f1aaa01a9734ccd) |
| 转换配方 | `qwen3_6_35b_a3b` |
| 转换器仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时 revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |

产物标识、对象清单摘要与转换溯源发布于
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/blob/main/artifact-manifest.json)。
精确的存储契约维护在
[v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/weights-and-loading.md) 中。

## 许可证

本 NInfer 产物按 Apache License 2.0 分发。源
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 仓库同样采用 Apache-2.0。
用户仍需自行遵守许可证及适用法律法规。
