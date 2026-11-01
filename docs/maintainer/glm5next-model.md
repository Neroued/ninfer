# GLM5Next 架构

GLM-5.3-Flash 的原生架构为 `Glm5NextForCausalLM / glm5next_text`。
它组合 KDA、Kpool Sparse MLA、mHC、dense/MoE、MTP 与独立 Vision tower。
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

W_A 将 latent、index-K、pool gate 作为连续 history 前缀，其后是 Q_a 与 head coefficients。
W_B 的 MLA-Q 位于 index-Q 之前。由此完整计算、selection 复用和 history-only 补齐可以借用同一
权重的不同连续前缀，不增加驻留副本或多次小投影。U_K/U_V 保留因子形式，转换时拆分源 KV_b。

## 表示选择

| 角色 | GLM-5.3-Flash recipe |
|---|---|
| Main routed experts、前三层 dense MLP | NVIDIA NVFP4，允许 A4 |
| KDA Q/K/V/O、MLA O、shared GU/down | Q8 A16 |
| MTP routed experts | 从源 BF16 转为 block-FP8，允许 A8 |
| KDA conv/control 的源 FP32 字段 | 保留 FP32 words |
| 其余投影、norm、mHC、embedding、Main head、Vision | 保留 BF16 |
| Optimized draft head | 额外完整词表 Q4 A16，Main head 保持 BF16 |

源 NVFP4 保留 codes/scales 与校准含义；GU 合并须满足原 divisor 关系。
转换与导入规则归 [converter](../../tools/convert/glm5next.py)和
[公共权重机制](weights-and-loading.md)，模型文档不重复 encoder。

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
| Vision | 权重复制，按完整 attention segment 分工 |

两侧共同持有每个请求的 KDA State，MLA Main/MTP 共用 attention owner。
State variant 使用公共字段前缀和 owner-only raw tail；checkpoint 必须同时覆盖 KDA shards 与
owner 上的 histories。ADP 不把单请求历史分成两半，它减少重复副本并让不同请求使用不同池容量。

MLA 边界按 owner-local 行 pack，输出汇集回全局顺序，再进入复制的 mHC 或 MTP residual。
零有效行也遵守通信次序。放置与压力转换使用[共享 Context 策略](context-management.md#attention-dp)，
字段转换和切片由 [ParallelPlan](../../src/models/glm5next/parallel/plan.cpp)及 Session 解释。

## MTP 与 Graph

MTP 是普通 residual 的 MLA＋MoE 层，没有 Main mHC/KDA；conditioning 为 Main final RMSNorm 后的
hidden。Stem 拼接归一化 embedding 与 conditioning，ordinal 0 的 embedding 分量置零。
其媒体输入使用词表 embedding，Main hidden 已携带视觉内容；Main 自己仍替换媒体 embedding。

配置启用 selection reuse 时，draft 首步产生选择与可见边界，后续步骤复用冻结结果。
它们仍写自己的 provisional history，不能随着后续位置自动扩张首步 tail 的选择范围。
最终采用先复用合法首步，再按真实 Main conditioning 补齐 canonical history；递归 hidden
不能仅因 proposal 被接受就当成真实 Main hidden。

Decode 主体按全局容量 G 和各 rank 共同的本地 attention 容量 C 组织，owner 分布通过设备行映射
进入，不枚举 owner 向量。采样参数、实际 draft/accepted 长度是输入数据；Indexer/MLA 局部 child
随上下文更新，外层通信保持固定。Decision 与最终 commit 依赖独立，Prefill/Replay 使用 eager。
共享原则见[Speculative 执行](speculative-decoding.md)与[分布式执行](distributed-execution.md)。

## Vision 与 Frontend

Vision 使用 patch projection、带 Q/K RMSNorm 与二维 RoPE 的 transformer、2×2 downsample 和
merger。它没有 Text 的生成期 KV。Converter 按 processor 的 patch/merge 次序保值重排权重，
直接以连续 view 完成 downsample 输入，不在运行期做整份布局转换。

视频按独立 temporal-grid attention segment 分工；完整 features 按原媒体顺序交给 Text。
CPU processor、输入 token 展开与 GPU 计算共用几何事实，执行窗口受[公共媒体预算](frontend.md#有界媒体准备)约束。

GLM Frontend 负责自身角色标记、thinking 与工具语言，复用公共 tokenizer/Jinja、媒体资源、
grammar 和输出事务。模型家族差异留在 Frontend，HTTP 层只消费公共事件。
