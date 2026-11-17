---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: mit
base_model:
- zai-org/GLM-5.3-Flash
- nvidia/GLM-5.3-Flash-NVFP4
base_model_relation: quantized
tags:
- ninfer
- glm5next
- nvfp4
- q8
- mtp
- blackwell
- dgx-spark
- multimodal
- conversational
- cuda
---

# NInfer 版 GLM-5.3-Flash

[English](README.md) | 简体中文

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) 的混合精度 NInfer v3 产物，基于
[NVIDIA 的 NVFP4 checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) 转换而来。
包含 Text、Vision、原生 MTP 以及一个额外的全词表 Q4 draft head。

原生 NInfer 架构为 GLM5Next，结合 KDA 线性注意力、Kpool 稀疏 MLA、mHC 与 MoE。产物内含源分词器、
GLM 对话模板和处理器资源。可直接配合 [NInfer](https://github.com/Neroued/ninfer) 使用，在
**两台 DGX Spark** 上进行 CLI 生成或兼容 OpenAI / Anthropic 的服务。

## 下载

```bash
hf download neroued/GLM-5.3-Flash-NInfer \
  --include 'glm5_3_flash.ninfer*' \
  --local-dir models/GLM-5.3-Flash-NInfer
```

完整产物共 6 个文件、**178.33 GiB**：

```text
glm5_3_flash.ninfer
glm5_3_flash.ninfer.part-0001
glm5_3_flash.ninfer.part-0002
glm5_3_flash.ninfer.part-0003
glm5_3_flash.ninfer.part-0004
glm5_3_flash.ninfer.part-0005
```

请将所有分片与入口文件放在同一目录，并在每个节点上都放置完整的一套。只把
`glm5_3_flash.ninfer` 传给 NInfer；加载时会按 rank 切分权重。文件大小与产物标识记录在
[artifact-manifest.json](artifact-manifest.json)。

## 权重表示

| 权重 | 存储表示 |
|---|---|
| 主路由专家与前三个稠密 MLP 层 | 导入的 NVIDIA NVFP4，带校准后的 scale |
| KDA Q/K/V/O、MLA 输出与共享专家投影 | Q8 A16 |
| MTP 路由专家 | 由源 BF16 转换的 Block-FP8 |
| 主 head、embedding、mHC、Vision 及其他保留参数 | BF16；源 FP32 KDA/控制字段仍为 FP32 |
| 优化后的 MTP head | 额外的全词表 Q4 A16；主 head 保持 BF16 |

[`glm5next` 配方](https://github.com/Neroued/ninfer/blob/master/tools/convert/official_recipes.py)
保留导入的主 NVFP4 codes/scales，并执行上述额外量化。优化后的 draft head 不裁剪词表。

## 在双 Spark 上运行

在两个节点上构建当前 NInfer 版本，需安装 NCCL 开发依赖：

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

完整的依赖与检出说明见[构建指南](https://github.com/Neroued/ninfer#build)。在仓库根目录准备部署配置：

```bash
cp examples/configs/spark2-glm5_3-flash.toml ninfer.local.toml
# 修改 coordinator/worker 的产物路径、worker 可执行文件路径和网络地址。
./build/dgx-spark-release/apps/ninfer-serve --config ninfer.local.toml
```

该[示例](https://github.com/Neroued/ninfer/blob/master/examples/configs/spark2-glm5_3-flash.toml)
使用 TP2 加注意力 DP2、FP8 KV、Vision、MTP3、优化后的 head、CUDA Graph 解码，最多 8 个活跃请求。
其逻辑上下文上限为 **1,048,576 tokens**。coordinator 通过 SSH 连接远端节点，并启动两个 worker。
客户端只需连接它唯一的 HTTP 端点。

专家默认使用 TP（`ep = 1`）；设置 `ep = 2` 可选用整专家 EP，同时其余部分仍保持 TP2 与注意力 DP2。
每个节点都需保留全部 6 个产物文件。完整产物无法放进单台 Spark。运行时分配还包括 workspace 与
共享 KV/State 池；请阅读所选配置在启动时给出的容量估算。

CLI 用法：在 coordinator 上运行
`apps/ninfer --config ninfer.local.toml --prompt "..." --max-tokens 2048`。
详细的启动与网络说明见
[部署指南](https://github.com/Neroued/ninfer/tree/master/examples/configs)。

## 模型选项

- Thinking 始终保持启用。支持的推理 effort 为 `low`、`high` 和 `max`；默认值由模板提供。
  `--no-thinking` 与 effort `none` 均不受支持。
- 原生 MTP 支持 1..5 个 draft token。`--proposal-head optimized` 选用 Q4 draft head；
  目标 head 保持 BF16。草稿数量可按负载与并发度调整。
- 潜在 KV 使用 `--kv-dtype bf16` 或 `fp8`。Kpool indexer 默认使用 FP8；
  `NINFER_KPOOL_INDEXER_BF16=1` 可独立选择 BF16 indexer 存储。
- Vision、工具调用、结构化输出、流式输出与前缀复用，和文本生成共用同一套公开 Engine 与协议端点。
  图像/视频输入需在启动时启用 Vision。
- GLM 使用自己的对话模板与工具封装格式；请保留随附的 GLM 前端资源。

该架构不使用 DFlash 和 DFlash2。NInfer 将生成的工具调用返回给客户端，不负责执行。

## 来源与许可证

主要转换源为
[nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)，revision
`da920bb0b9f4a06727223a349e55468e38352348`，基于
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)。
模型权重采用上游的 [MIT 许可证](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)。
[NInfer](https://github.com/Neroued/ninfer) 软件采用 Apache-2.0 许可证。
