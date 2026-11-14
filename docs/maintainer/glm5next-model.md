# GLM5Next 架构

GLM-5.3-Flash 的原生架构为 `Glm5NextForCausalLM / glm5next_text`。
它组合 KDA、Kpool Sparse MLA、mHC、dense/MoE、独立 Vision tower，以及可选 MTP 或 DFlash2。
Config 决定实例层序与维度，Op 决定其可执行范围。完整产物的部署使用双 Spark。

定义与计算入口：[config](../../src/models/glm5next/config.cpp)、
[load](../../src/models/glm5next/load/)、[execution](../../src/models/glm5next/execution/)。

## 主干

```text
embedding / Vision features → 四路 residual
  → attention mHC pre → KDA 或 Sparse MLA → mHC post
  → FFN mHC pre → dense 或 routed + shared MoE → mHC post
  → 四流均值 → final RMSNorm → Main head
```

mHC 通过小投影产生 pre/post 与 mixing 系数，Sinkhorn 归一化 mixing，再形成分支输入与 residual。
融合相邻 post/pre 时仍消费已公开的 BF16 residual，不能写出 BF16 一份却用未舍入值生成下一组
系数。其余未形成公共表示的中间值不人为增加低精度舍入。精确数学由 [mHC Op](../../include/ninfer/ops/mhc.h)维护。

Text norm 使用直接 gamma。FFN 使用 clamped SwiGLU，clamp 位于 SiLU/product 前。
Router 的 correction bias 只改变选择，最终系数来自未加 bias 的 sigmoid scores；shared 直接
加入，不使用 Qwen 的独立 shared sigmoid gate。实例的常量和几何由配置与参数绑定取得。

## KDA 与 ReplaySSM

KDA 的 Q/K/V 经因果卷积和 SiLU；forget/output gate 通过各自低秩投影产生。
它使用逐 key 维的 decay，recurrent State 为 FP32，卷积 history 为 BF16。
长 Prefill 采用完整更新，短 speculative verify 保存 transition inputs，接受后 fold。

Record 固定同次执行实际消费的 represented inputs；fold 重放相同有限精度 transition。
Chunked Prefill 的数学精度与 recurrent record/fold 的精确关系分别验证，不能要求不同算法逐位相同。
公共设计见[Speculative 执行](speculative-decoding.md#replayssm)，接口见
[KDA](../../include/ninfer/ops/kimi_delta_attention.h)和
[fold](../../include/ninfer/ops/kda_replay.h)。

## Kpool Sparse MLA

本实例 Text MLA 为 NoPE。Prepare 同时形成 normalized latent、raw index K/gate 和 query
所需投影；每四 token 使用 learned per-channel weights 压缩 index key。
Indexer 以带符号 head coefficients 加权非负点积，选择完整 pools；所有 MLA heads 共用选择。
未完成 pool 的可见 tail 单独参与 attention，index score 不加入 attention logits。

MLA 保存逐 token latent，不保存展开的全 head K/V。U_K 将 query 映射到 latent 空间，
attention 后 U_V 映射回 head 输出，再执行 O projection。Indexer 历史和 latent 共享原 token
coverage，raw K/gate tail 属于 State。实现合同见
[Kpool MLA](../../include/ninfer/ops/kpool_mla.h)及其
[布局](../../include/ninfer/ops/kpool_mla_layout.h)。

Prepare 按逻辑角色绑定 latent、index-K、pool gate、Q_a 与 head coefficients。
`glm5next` recipe 使用连续 history 前缀；FP4 Local 按源表示组织多个投影组，避免为合并而改变
编码。完整计算、selection 复用和 history-only 补齐各自只消费所需行，借用原 parent backing。
U_K/U_V 保留因子形式，转换时拆分源 KV_b；实际连续分组由绑定和 recipe 表达。

## 表示选择

| 角色 | `glm5next` | `glm5next_fp4_local` |
|---|---|---|
| Main routed experts、dense MLP | NVIDIA NVFP4 divisor 表示，允许 A4 | 保留源 `nvfp4_mul`、CSF 与校准 multiplier，允许 A4 |
| KDA 大投影、MLA O、shared GU/down | Q8 A16 | 保留各自源编码；MXFP8 使用 AllowA8 |
| MTP routed experts | 源 BF16 转 block-FP8，允许 A8 | 保留源 NVFP4 multiplier 表示，A16Only |
| 其余参数 | 保留 BF16 或源 FP32 | 保留源 BF16、FP32、MXFP8 或带 FP32 scales 的 block-FP8 |

两种 recipe 使用同一架构和执行入口。源编码及 activation 辅助值由
[converter](../../tools/convert/glm5next.py)与 [recipe](../../tools/convert/official_recipes.py)绑定；
CSF、乘除 scale 和分片物化的公共关系见[权重与加载](weights-and-loading.md)。

DFlash2 自身的参数保留 companion 的原始浮点表示。Embedding 和 full proposal head 共享当前
Main 绑定，因而 full 不表示强制 BF16。`--proposal` 另生成完整词表 Q4 A16 head，MTP 与
DFlash2 均可选择它；这不改变 target head 或 DFlash2 decoder 权重。

Latent 使用已有 `kv_dtype=bf16|fp8`。Indexer 默认 FP8，
`NINFER_KPOOL_INDEXER_BF16=1` 在启动选择 BF16 并传给各 rank，不另设 Engine KV 选项。
FP8 cache 共用 E4M3FN/G128/UE8M0 codec；latent 保留原始基底，index keys 使用归一化 H128 基底。
Raw tail/records 保留 BF16。Prepare 的 FP8 编码消费 FP32 norm/pooling 结果，不先舍入到 BF16。
具体计算边界由 Op 合同维护，格式压缩比不等于端到端收益。

## 并行与放置

| 部分 | 双 Spark 分工 |
|---|---|
| KDA | Heads TP2，conv/recurrent 分片 |
| MLA / Indexer | 请求 ADP2；owner 处理完整 attention 并保存该请求的 latent/index history |
| Dense / shared FFN | Intermediate TP2 |
| Routed experts | TP2/EP1 为 expert TP；TP2/EP2 为完整 expert EP |
| mHC / residual、embedding、MTP stem | 复制 |
| Main / draft head | 词表 TP2 |
| DFlash2 | Attention heads、FFN intermediate 与 feature projection 按 TP2 分片；其余小型参数复制 |
| Vision | 权重复制，按完整 attention segment 分工 |

两侧共同持有每个请求的 KDA State，MLA Main/MTP 共用 attention owner。
State variant 使用公共字段前缀和 owner-only raw tail；checkpoint 必须同时覆盖 KDA shards 与
owner 上的 histories。ADP 不把单请求历史分成两半，它减少重复副本并让不同请求使用不同池容量。
DFlash2 使用自己的 GQA 与滑动窗口 context；context KV 按 heads TP 分片并存入定长 State。
Checkpoint 同时覆盖 Main 和当前启用 backend 的续算状态。

MLA 边界按 owner-local 行 pack，输出汇集回全局顺序，再进入复制的 mHC 或 MTP residual。
零有效行也遵守通信次序。放置与压力转换使用[共享 Context 策略](context-management.md#attention-dp)，
字段转换和切片由 [ParallelPlan](../../src/models/glm5next/parallel/plan.cpp)及 Session 解释。

## MTP

MTP 是普通 residual 的 MLA＋MoE 层，没有 Main mHC/KDA；conditioning 为 Main final RMSNorm 后的
hidden。Stem 拼接归一化 embedding 与 conditioning，ordinal 0 的 embedding 分量置零。
其媒体输入使用词表 embedding，Main hidden 已携带视觉内容；Main 自己仍替换媒体 embedding。

配置启用 selection reuse 时，draft 首步产生选择与可见边界，后续步骤复用冻结结果。
它们仍写自己的 provisional history，不能随着后续位置自动扩张首步 tail 的选择范围。
最终采用先复用合法首步，再按真实 Main conditioning 补齐 canonical history；递归 hidden
不能仅因 proposal 被接受就当成真实 Main hidden。

## DFlash2

DFlash2 从配置指定 Main 层的特征建立 conditioning，经 feature projection 与 norm 物化各层
context K/V。草拟输入为 anchor 与 masked block，经过 GQA、动态卷积和 FFN，再用词表候选及
条件 selector 生成 draft 路径和实际 proposal 分布。Target 的 KDA/MLA 与接受过程保持原语义。

每层的 K/V/Q 使用合并 parent，context-only 路径借用 K/V 前缀。特征、候选分布及 provisional
计算属于执行帧；正式 context 只采用已提交输入。草拟窗口与验证有效长度分别管理，
记录和状态采用遵循[Speculative 执行](speculative-decoding.md)。

MTP 与 DFlash2 是互斥的启动选择，只加载所选 backend、head 和执行资源。
组合入口见 [DFlash2 execution](../../src/models/glm5next/execution/dflash.cpp)与
[Program](../../src/models/glm5next/program/dflash.cpp)。

## Graph

Decode 主体按全局物理桶 P 和各 rank 共同的本地 Attention 桶 L 组织，owner 分布通过设备行映射
进入，不枚举 owner 向量。采样参数、实际 draft/accepted 长度是输入数据；Indexer/MLA 局部 child
随上下文更新，外层通信保持固定。Decision 与最终 commit 依赖独立，Prefill/Replay 使用 eager。
批次与 padding 合同见[并发执行](concurrent-execution.md)，其他共享原则见
[Speculative 执行](speculative-decoding.md)与[分布式执行](distributed-execution.md)。

## Vision 与 Frontend

Vision 使用 patch projection、带 Q/K RMSNorm 与二维 RoPE 的 transformer、2×2 downsample 和
merger。它没有 Text 的生成期 KV。Converter 按 processor 的 patch/merge 次序保值重排权重，
直接以连续 view 完成 downsample 输入，不在运行期做整份布局转换。

视频按独立 temporal-grid attention segment 分工；完整 features 按原媒体顺序交给 Text。
CPU processor、输入 token 展开与 GPU 计算共用几何事实，执行窗口受[公共媒体预算](frontend.md#有界媒体准备)约束。

GLM Frontend 负责自身角色标记、thinking 与工具语言，复用公共 tokenizer/Jinja、媒体资源、
grammar 和输出事务。模型家族差异留在 Frontend，HTTP 层只消费公共事件。
