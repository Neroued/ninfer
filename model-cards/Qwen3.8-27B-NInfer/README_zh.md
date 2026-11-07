---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.67
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
            value: 87.37
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
            value: 82.22
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# NInfer 版 Qwen3.8-27B

[English](README.md) | 简体中文

本模型卡是 [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
的版本控制源文件。

本仓库包含转换为 [NInfer](https://github.com/Neroued/ninfer) 原生 `.ninfer` 产物格式的
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。该产物仅供 NInfer 使用，
不是 Transformers checkpoint、Safetensors 分发包或 GGUF 文件。

## 产物

| 字段 | 值 |
|---|---|
| 文件名 | `qwen3_8_27b.ninfer` |
| 大小 | 20,437,521,664 bytes (19.03 GiB) |
| SHA-256 | `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da` |
| 容器版本 | 3 |
| 架构 | `Qwen3_5ForCausalLM` |
| 公开模型名 | `qwen3.8-27b` |
| 对话模板 | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja); 可用 `--chat-template FILE` 覆盖 |
| 模板默认值 | 启用 thinking；effort `xhigh`；保留已结束轮次的推理内容 |
| 存储对象数 | 1,190（1,184 个张量和 6 个资源） |

Text 主体使用 Q4/Q5 投影，token embedding 与完整输出 head 使用 `q8_g32_fp16`。
文件还包含 Vision、MTP、DFlash2、优化后的 proposal head 和前端资源。
Vision 与投机解码权重仅在启动时选用后才加载。

用以下命令校验下载文件：

```bash
printf '%s  %s\n' \
  '81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da' \
  'qwen3_8_27b.ninfer' | sha256sum --check
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
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

./build/rtx5090-release/apps/ninfer models/qwen3_8_27b.ninfer \
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
./build/rtx5090-release/apps/ninfer-serve models/qwen3_8_27b.ninfer \
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
- 使用随附 DFlash2 配套权重的 DFlash2，草稿窗口 1 至 15
  （`--spec dflash2 --draft-tokens 7`，可选 `--proposal-head optimized`）；
- BF16、INT8、FP8、NVFP4 和 K8V4 KV 缓存；
- CUDA Graph 解码与兼容前缀复用；
- 启动时定容的小规模并发服务，支持真正的批量解码；
- NInfer CLI；
- OpenAI Responses Core、OpenAI Chat Completions 和 Anthropic Messages 服务。

## 评测

该产物通过 NInfer 兼容 OpenAI 的服务路径评测，启用 thinking、MTP=3 和 INT8 group-64 KV。
EvalScope 1.9.0 使用 0-shot prompt、规则评分、每题一次采样，参数为 temperature 1.0、
top-p 0.95、top-k 20、presence penalty 0.0、seed 42。文本测试集使用 262,144-token
上下文上限；多模态测试集使用 `--vision` 和 81,920-token 上限。

| 基准 | NInfer groupwise-int | 正确数 / 总数 | 官方 Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench (prompt-level strict) | 77.67% | 233 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 87.37% | 173 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 82.22% | 629 / 765 | 85.9 |

全部 1,723 个配置样本均完成并计分。IFBench 另报告 81.00% instruction-level strict、
80.67% prompt-level loose 和 83.67% instruction-level loose。
这些是单样本结果，不是 pass@k。

官方 Qwen3.8-27B BF16 数据来自
[上游模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)；其中未说明采样设置与 IFBench
指标层级，因此最后一列并非同一协议下的对比。在 4 个重叠基准上，groupwise-int 的差值
保持在 ±3.1 个百分点以内，且上游模型卡未报告 AIME 结果。

## 限制

- 该产物面向单张 RTX 5090，每个 Engine 支持 1 至 8 个活跃请求。
- KV 与续写状态共享启动时创建的 Context 池。可用上下文取决于所选 KV 类型、
  启用的组件、并发度和可用内存。
- NInfer 将生成的工具调用返回给客户端，不负责执行。

## 溯源

| 字段 | 值 |
|---|---|
| 源仓库 | `Qwen/Qwen3.8-27B` |
| 源 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| 下载源 | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| 转换配方 | `qwen3_8_27b` |
| 转换器仓库 | `https://github.com/Neroued/ninfer` |
| 最低运行时 revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |
| Ranking 输入 SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

在发布转换产物前，已对照上述规范的 Hugging Face 源 revision 核对本地源配置、张量索引、
前端资源与已发布的 CRC32 清单。

产物标识、对象清单摘要与转换溯源发布于
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json)。
精确的存储契约维护在
[v3 容器参考](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/weights-and-loading.md) 中。

## 许可证

本 NInfer 产物按 Apache License 2.0 分发。源
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 仓库同样采用 Apache-2.0。
用户仍需自行遵守许可证及适用法律法规。
