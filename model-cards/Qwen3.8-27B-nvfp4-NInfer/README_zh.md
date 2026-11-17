---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.00
            name: Prompt-level strict (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 96.67
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
            value: 96.67
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
            value: 90.40
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: ERQA
          type: erqa
        metrics:
          - type: accuracy
            value: 66.25
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: RealWorldQA
          type: real_world_qa
        metrics:
          - type: accuracy
            value: 83.53
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# NInfer 版 Qwen3.8-27B NVFP4

[English](README.md) | 简体中文

本模型卡是
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
的版本控制源文件。

本仓库包含 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的混合 NVFP4/FP8 表示。
它把官方 BF16 checkpoint 与
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
中固定的打包 Text 权重结合为 [NInfer](https://github.com/Neroued/ninfer) 原生 `.ninfer`
产物格式。该产物仅供 NInfer 使用，不是 Transformers checkpoint、Safetensors 分发包或 GGUF 文件。

该产物使用 Qwen3.5 Dense 架构。Text 第 0–55 层使用 NVFP4 MLP 权重；token embedding、
attention input/output 投影、GDN Q/K/V/Z 与 output 投影、完整输出 head，以及 Text 第 56–63 层
MLP 权重使用 row-scaled FP8。控制权重使用 BF16，MTP、Vision 与 DFlash2 权重则独立存放。

## 产物

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_8_27b_nvfp4.ninfer` |
| 大小 | 23,719,715,844 bytes (22.09 GiB) |
| SHA-256 | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5ForCausalLM` |
| 公开模型名 | `qwen3.8-27b` |
| 对话模板 | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja); 可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 启用 thinking；effort `xhigh`；保留已结束轮次的推理内容 |
| 存储对象数 | 1,246（1,240 个张量和 6 个资源） |
| NVFP4 张量 | 112 |
| Row-scaled FP8 张量 | 146 |

文件内含 Text、Vision、MTP、DFlash2、优化后的 proposal head 和前端资源。
Vision 与投机解码权重仅在启动时选用后才加载。源自上游的 NVFP4 与 FP8 字保留原样，
不做解码和重量化；只有官方 BF16 token embedding 在本地编码为 row-scaled FP8。

用以下命令校验下载文件：

```bash
printf '%s  %s\n' \
  '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82' \
  'qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

本版本包含来自 `z-lab/Qwen3.8-27B-DFlash2`、revision 为
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4` 的完整 DFlash2 配套权重。
选用 `--spec dflash2 --draft-tokens 7 --proposal-head optimized`；支持 1..15 个 draft token。
DFlash2 请使用当前运行时。已有的性能与评测表格保留其注明的 MTP 配置与 revision。

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
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
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

在 NInfer 仓库根目录，复制
[RTX 5090 部署示例](https://github.com/Neroued/ninfer/blob/master/examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml)
并把 `model.path` 指向下载好的产物：

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
# 修改 model.path，指向你的产物。
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

该示例启用 Vision、FP8 KV、CUDA Graph 解码、MTP3 和优化后的 proposal head，
配置 2 个活跃请求、1,024-token prefill 分块和 256 MiB 内存余量。
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
- 使用随附 DFlash2 配套权重的 DFlash2，草稿窗口 1 至 15
  （`--spec dflash2 --draft-tokens 7`，可选 `--proposal-head optimized`）；
- BF16、INT8、FP8、NVFP4 和 K8V4 KV 缓存；
- CUDA Graph 解码与兼容前缀复用；
- 启动时定容的小规模并发服务，支持真正的批量解码；
- NInfer CLI；
- OpenAI Responses Core、OpenAI Chat Completions 和 Anthropic Messages 服务。

## 性能

测量于 2026 年 9 月 28–29 日，NInfer revision 为
[`7f6aafed`](https://github.com/Neroued/ninfer/commit/7f6aafedb5f20200def820cfe51ab81c09c20eeb)，
使用单张 RTX 5090、驱动 617.14，以及 CUDA 13.4 编译/运行时/driver API。
这些服务测试使用 FP8 E4M3 row-256 KV、CUDA Graphs、1,024-token prefill 分块、关闭前缀复用，
以及 temperature 0.6 / top-p 0.95 / top-k 20 / min-p 0 / presence penalty 1.0 /
frequency penalty 0。MTP0 的上下文上限为 262,144 tokens；MTP3 每请求使用 131,072 tokens、
3 个 draft token、优化后的 proposal head 和自动确定的共享 KV 容量。

### 并发 MTP=3 语料 makespan

每个 C 运行一份完整的 75 请求语料，含 3 个推理 fixture 和 12 个跨场景 fixture，
每个 fixture 5 个种子，发送顺序固定打乱。Makespan 包含 prefill、解码、准入等待、
切换和排空；实际输出长度各不相同。

| C | 请求数 | 计算的 prefill token 数 | 解码 token 数 | Makespan (s) | 请求数/s | 语料 prefill (tok/s) | 语料解码 (tok/s) | 平均批大小 | MTP 接受率 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 693,701 | 4,115.22 | 0.0182 | 3.8 | 168.6 | 1.00 | 59.6% |
| 2 | 75 | 15,460 | 709,989 | 2,394.23 | 0.0313 | 6.5 | 296.5 | 1.90 | 58.9% |
| 4 | 75 | 15,460 | 722,202 | 1,640.17 | 0.0457 | 9.4 | 440.3 | 3.13 | 58.5% |
| 8 | 75 | 15,460 | 708,589 | 1,443.37 | 0.0520 | 10.7 | 490.9 | 3.52 | 59.5% |

全部 300 个请求均完成，未出现请求、CUDA 或分配错误。在 C=1/2/4/8 时，自动 KV 容量为
131,072 / 262,144 / 253,632 / 225,024 tokens。C=8 的平均批大小为 3.52，
采样区间内最多有 5 个等待请求。常驻 MTP3 权重占用 19.729 GiB，workspace arena 为 243.3 MiB。

### 长上下文服务（关闭 MTP）

数值为每个 fixture 下 5 个固定种子的算术平均 ± 样本标准差。

| Prompt tokens | 样本数 | Prefill 阶段 (tok/s) | 服务端 TTFT (ms) | 解码阶段 (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 12,819.1 ± 16.8 | 602.7 ± 1.1 | 74.1 ± 0.3 |
| 64,512 | 5 | 8,658.2 ± 44.8 | 7,487.4 ± 37.7 | 68.2 ± 0.2 |
| 130,048 | 5 | 6,198.5 ± 19.6 | 21,055.7 ± 67.2 | 62.5 ± 0.3 |
| 260,096 | 5 | 4,016.4 ± 10.4 | 64,910.0 ± 171.8 | 53.4 ± 0.6 |

### MTP=3 单请求长推理解码

这些阶段统计来自 C=1 语料，每个推理 fixture 5 个样本。

| Fixture | 样本数 | 完成 token 数 | 解码阶段 (tok/s) | MTP3 接受率 | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,559.4 ± 727.2 | 207.4 ± 3.6 | 77.2% ± 1.8% | 3.32 ± 0.05 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 161.7 ± 3.8 | 57.2% ± 2.1% | 2.72 ± 0.06 |
| `long_decode_aime26_30` | 5 | 37,978.0 ± 7,474.4 | 170.0 ± 2.3 | 59.9% ± 1.5% | 2.80 ± 0.05 |

### MTP=3 单请求跨场景解码

每个类别汇总 3 个 fixture × 5 个种子。数值为平均 ± 样本标准差。

| 类别 | 样本数 | 解码阶段 (tok/s) | MTP3 接受率 | MTP3 tokens/round |
|---|---|---|---|---|
| 代码 | 15 | 205.3 ± 9.5 | 76.8% ± 5.1% | 3.30 ± 0.15 |
| 故事 | 15 | 132.6 ± 13.1 | 37.6% ± 7.1% | 2.13 ± 0.21 |
| 翻译 | 15 | 202.9 ± 12.3 | 75.2% ± 6.6% | 3.26 ± 0.20 |
| 结构化 | 15 | 231.7 ± 10.6 | 90.6% ± 5.6% | 3.72 ± 0.17 |

AIME 15 的全部 5 个样本都达到 65,536-token 输出预算。C=1 语料包含 40 个 stop-token 结果
和 35 个达到输出上限的结果；全部保留在统计中。这些测量不对答案准确率或任务完成度评分。

[完整结果与复现命令](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md)
还涵盖 DFlash2 K=7、MTP3 解码饱和测试和完成结果。

## 评测

该产物通过 NInfer 兼容 OpenAI 的服务路径评测，启用 thinking、MTP=3 和 INT8 group-64 KV。
EvalScope 1.9.0 使用 0-shot prompt、规则评分、每题一次采样，参数为 temperature 1.0、
top-p 0.95、top-k 20、presence penalty 0.0、seed 42。文本测试集使用 252,928-token
上下文上限；多模态测试集使用 `--vision` 和 81,920-token 上限。

| 基准 | NInfer NVFP4 | 正确数 / 总数 | 官方 Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench (prompt-level strict) | 77.00% | 231 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 90.40% | 179 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 83.53% | 639 / 765 | 85.9 |

全部 1,723 个配置样本均完成并计分。IFBench 另报告 80.50% instruction-level strict、
80.33% prompt-level loose 和 83.50% instruction-level loose。
这些是单样本结果，不是 pass@k。

官方 Qwen3.8-27B BF16 数据来自
[上游模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)；其中未说明采样设置与 IFBench
指标层级，因此最后一列并非同一协议下的对比。在 4 个重叠基准上，NVFP4 的差值保持在
±2.5 个百分点以内，且上游模型卡未报告 AIME 结果。

## 限制

- 该产物面向单张 RTX 5090，每个 Engine 支持 1 至 8 个活跃请求。
- KV 与续写状态共享启动时创建的 Context 池。可用上下文取决于所选 KV 类型、
  启用的组件、并发度和可用内存。
- NInfer 将生成的工具调用返回给客户端，不负责执行。

## 溯源

| 字段 | 值 |
|---|---|
| 基础仓库 | `Qwen/Qwen3.8-27B` |
| 基础 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| 基础下载源 | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| 量化源仓库 | `unsloth/Qwen3.8-27B-NVFP4` |
| 量化源 revision | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| 转换配方 | `qwen3_8_27b_nvfp4` |
| Embedding 编码器 | `fp8_row_maxabs` |
| 转换器仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时 revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |
| Ranking 输入 SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

产物标识、对象清单摘要与转换溯源发布于
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json)。
精确的存储契约维护在
[v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/weights-and-loading.md) 中。

## 许可证

本 NInfer 产物按 Apache License 2.0 分发。
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 基础仓库和
[量化源仓库](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 同样采用 Apache-2.0。
用户仍需自行遵守许可证及适用法律法规。
