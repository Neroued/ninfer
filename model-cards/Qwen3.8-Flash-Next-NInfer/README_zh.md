---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
- Qwen/Qwen3.8-Flash-Next
- nvidia/Qwen3.8-Flash-Next-NVFP4
base_model_relation: quantized
tags:
- ninfer
- qwen4-exp
- nvfp4
- q8
- nf4
- mtp
- blackwell
- dgx-spark
- multimodal
- conversational
- cuda
model-index:
- name: Qwen3.8-Flash-Next-NInfer
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA-Diamond
      type: gpqa_diamond
    metrics:
    - type: accuracy
      value: 91.79
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: AIME 2026
      type: aime26
    metrics:
    - type: accuracy
      value: 97.78
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: ERQA
      type: erqa
    metrics:
    - type: accuracy
      value: 68.0
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: RealWorldQA
      type: real_world_qa
    metrics:
    - type: accuracy
      value: 85.75
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: MMMU-Pro (standard, 10 options)
      type: mmmu_pro
    metrics:
    - type: accuracy
      value: 80.81
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
---

# NInfer 版 Qwen3.8-Flash-Next

[English](README.md) | 简体中文

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的混合精度 NInfer v3 产物，
把 BF16 checkpoint 与
[NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
中经过校准的专家权重和 MTP 权重结合。内含 Text、Vision、原生 MTP 和优化后的全词表 draft head。

可配合 [NInfer](https://github.com/Neroued/ninfer) 用于 CLI 生成或兼容 OpenAI / Anthropic 的服务。
原生架构为 Qwen4Exp：Gated DeltaNet、稀疏注意力、门控残差流、MoE 与 PLE embedding。产物自带
分词器、处理器资源和维护中的 Flash Next 对话模板。`.ninfer` 文件由 NInfer 直接读取。

## 下载

```bash
hf download neroued/Qwen3.8-Flash-Next-NInfer \
  --include 'qwen3_8_flash_next.ninfer*' \
  --local-dir models/Qwen3.8-Flash-Next-NInfer
```

完整产物共 4 个文件、**101.00 GiB**：

```text
qwen3_8_flash_next.ninfer
qwen3_8_flash_next.ninfer.part-0001
qwen3_8_flash_next.ninfer.part-0002
qwen3_8_flash_next.ninfer.part-0003
```

请把 4 个文件放在同一目录，且只把 `qwen3_8_flash_next.ninfer` 传给 NInfer。这些是存储分片；
加载时会选择所需的 rank 切片。文件大小与产物标识记录在
[artifact-manifest.json](artifact-manifest.json)。

## 权重表示

| 权重 | 存储表示 |
|---|---|
| 主路由专家 | 导入的 NVIDIA NVFP4，带校准后的 scale |
| 大型 GDN 投影；主 QSA query/gate 与输出 | Q8 A16 |
| MTP 路由专家 | 导入的 block-FP8 |
| PLE 表 | NF4 |
| 主 head、共享专家、HC、Vision 及其他保留投影 | BF16；源 FP32 控制字段仍为 FP32 |
| 优化后的 MTP head | 额外的全词表 Q4 A16；主 head 保持 BF16 |

导入的 NVFP4 codes/scales 予以保留。额外量化由
[`qwen3_8_flash_next` 配方](https://github.com/Neroued/ninfer/blob/master/tools/convert/official_recipes.py)
完成。优化后的 head 不裁剪词表，可通过
`--spec mtp --draft-tokens 3 --proposal-head optimized` 选用。

## 在 Spark 上运行

按[构建说明](https://github.com/Neroued/ninfer#build) 构建当前 NInfer 检出版本。单台 Spark 使用
`dgx-spark-release` 预设。双 Spark 需在每个节点上构建同一版本，启用
`-DNINFER_ENABLE_DISTRIBUTED=ON` 并安装 NCCL 开发依赖。

在 NInfer 仓库根目录执行：

```bash
cp examples/configs/spark-qwen3_8-flash-next.toml ninfer.local.toml
# 把 model.path 指向下载好的入口文件。
./build/dgx-spark-release/apps/ninfer-serve --config ninfer.local.toml
```

双 Spark 请从
[`spark2-qwen3_8-flash-next.toml`](https://github.com/Neroued/ninfer/blob/master/examples/configs/spark2-qwen3_8-flash-next.toml)
开始，在两个节点上都放置全部分片，并修改 worker 路径、模型路径和网络设置。
服务端只在 coordinator 上运行一次，由它启动两个 worker。TP2 默认使用专家 TP；
`ep = 2` 选择整专家 EP。两个示例都启用 Vision、FP8 KV、MTP3、优化后的 head 和
CUDA Graph 解码，逻辑上下文上限为 262,144 tokens。

单次 CLI 请求使用 `apps/ninfer` 并传入相同配置和 `--prompt` 或 `--messages`。
命令见[部署指南](https://github.com/Neroued/ninfer/tree/master/examples/configs)。

RTX 5090 可通过 CPU 专家卸载运行该产物。该路径约需 63.3 GiB 的 Host 主专家后备内存，
外加其他运行时/系统内存。单台 Spark 上专家常驻 GPU，PLE 由磁盘支撑；双 Spark 则把 PLE
分片到设备内存。文件大小不等于运行时内存需求：KV/State、workspace 和启用的组件同样影响容量。

## 功能

- Text、图像/视频 prompt、流式输出、工具调用与结构化输出。
- 原生 MTP，支持 1..5 个 draft token，可选完整 head 或优化后的 head；用 `--vision` 启用 Vision。
- CUDA Graph 解码、前缀复用，最多 8 个活跃请求。
- CLI、OpenAI Chat/Responses 与 Anthropic Messages 共用同一 Engine。
- 默认启用 thinking。维护中的模板支持 `low`、`medium`、`xhigh` effort；
  `high`/`max` 映射到 `xhigh`，`minimal` 映射到 `low`，`none` 关闭 thinking。

上下文上限按请求计。实际共享 KV/State 容量在启动时报告，取决于部署方式与所选资源。
该架构不使用 DFlash 和 DFlash2。NInfer 把工具调用返回给客户端，不负责执行。

## 评测

记录于 **2026-10-21–22**，通过 NInfer 公开的兼容 OpenAI 服务和 EvalScope 1.10.0 完成。
服务端使用**两台 DGX Spark、TP2/EP2、BF16 KV、Vision、MTP3、优化后的 draft head、CUDA Graphs、
prefill 分块 4096、上下文上限 262,144、并发度 8**。

| 基准 | 准确率 | 正确数 / 计分样本数 | 每题重复次数 |
|---|---:|---:|---:|
| GPQA-Diamond | 91.79% | 727 / 792 | 4 |
| AIME 2026 | 97.78% | 88 / 90 | 3 |
| ERQA | 68.00% | 272 / 400 | 1 |
| RealWorldQA | 85.75% | 656 / 765 | 1 |
| MMMU-Pro (standard, 10 options) | 80.81% | 1,398 / 1,730 | 1 |

分数是生成样本的平均准确率，不是 pass@k。GPQA 含 198 题、每题 4 个样本；
AIME26 含 30 题、每题 3 个样本。其余数据集每题 1 个样本。MMMU-Pro 使用标准 10 选项格式。

所有任务均完成，无失败样本。2 条 GPQA 输出达到长度上限且没有解析出答案，
1 条 MMMU-Pro 输出未能解析；它们仍计入计分总数。原始数据集标签与评分器输出均已保留。
视觉结果是在后续 JPEG 解码器修复之前采集的。

生成使用 thinking 的 `xhigh` effort、temperature 1.0、top-p 0.95、top-k 20、min-p 0、
presence/frequency penalty 均为 0，输出预算 131,072 tokens。评测使用 0-shot prompt、
规则评分，评估器 seed 为 42。[evaluation.json](evaluation.json) 保存了运行标识、配置、
计数与诊断信息。

## 来源与许可证

- BF16 参数与前端资源：[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
- 校准后的专家与 MTP：[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)。
- 转换与运行时：[NInfer](https://github.com/Neroued/ninfer)。

源模型采用 [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)。
NVIDIA checkpoint 列出了 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
以及 Qwen 源条款。这些源权重条款同样适用于本衍生产物。NInfer 的 Apache-2.0
软件许可证与模型许可证相互独立。
