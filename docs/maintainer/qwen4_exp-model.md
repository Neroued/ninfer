# Qwen4Exp 架构

Qwen3.8-Flash-Next 使用 Qwen4Exp：GDN/QSA、四路 Gated Residual、MoE 与 PLE 的组合。
实例维度来自 config，逻辑权重来自绑定；不根据 checkpoint 名称选择另一套层实现。
主要入口为 [config](../../src/models/qwen4_exp/config.cpp)、
[load](../../src/models/qwen4_exp/load/)和 [execution](../../src/models/qwen4_exp/execution/)。

## 主干与状态

```text
embedding / Vision features → 四路 residual
  → attention HC → GDN 或 QSA → HC combine
  → FFN HC → routed + shared MoE → HC combine
  → final HC mixer → Main head
```

PLE 在配置指定层的 attention HC 前加入。HC 的 streams 是特征轴，不增加 GDN、QSA 或 MoE
的 token 数。Final mixer 的 sample hidden 与 mixer 前的四流 residual 是不同结果，
MTP conditioning 使用后者。

GDN 从 HC 输入形成 QKVZ 和 controls。Q/K/V 经过因果卷积与 SiLU，Z 绕过卷积；
递推输出经过直接 gamma 的 head norm 和 `sigmoid(Z)`，再投影回 hidden。
Conv 的 SiLU 与输出 gate 的 sigmoid 分属不同位置。

QSA 保存完整主 K/V；四 token 压缩只作用于 index key。完整块的 pooled key 经 norm/RoPE
参与打分，所有 query heads 共用 TopK 块集合；Sparse GQA 另消费未完成块的可见 tail。
Indexer score 不加入 attention logits。Raw index tail 与位置属于定长 State，完整历史属于 KV。

完整 checkpoint 还包含 GDN、PLE conv、continuation hidden 及启用 MTP 时所需的状态。
Raw token 账本与模型前沿归 Session。具体恢复、共享和分配沿用[Context 管理](context-management.md)。

## 权重选择与合并

Flash Next recipe 采用以下主要取舍；精确参数映射由
[converter](../../tools/convert/qwen4_exp.py)维护。

| 角色 | 选择 |
|---|---|
| Main routed experts | 导入 NVIDIA NVFP4，允许 A4 |
| GDN 大投影、Main QSA Q/gate 与 O | Q8 A16 |
| MTP routed experts | Block-FP8，允许 A8 |
| HC、控制投影、shared、主 head、Vision、其他保留投影 | BF16，保留对应数学边界 |
| PLE table | NF4，精确 BF16 lookup 出口 |
| Optimized MTP head | 从 Main 源 head 额外生成完整词表 Q4 A16，不裁词表 |

Offset norm 保留源 offset words，`1+w` 与 HC 缩放不提前折入低精度权重。
GDN/PLE convolution 转为 tap 外、channel 连续的布局，保持 tap 时间次序。
NVFP4 weight divisor 与 activation divisor 分属权重和 Use，见[权重与加载](weights-and-loading.md)。

合并投影保留逻辑角色。Main QSA 的不同表示分开；MTP 的 K、V、index-K 放在同一个 BF16 parent
的连续前缀，随后才是 Q、gate、index-Q。完整投影与 history-only 计算共享权重，避免补齐历史时
多次小 Linear。TP 先按角色切片再合并，不能把整个 QKV 类 parent 简单对半截断。
LinearSplit 直接写各段连续输出，模型不重复整理这些投影结果。

## PLE

PLE 根据 raw token 的二、三元组 hash 查表，组合 lookup embedding，再经投影、gate 与
带 dilation 的卷积作用于 residual。视觉特征不替换 hash 输入的 token IDs。
Lookup、投影和卷积有各自表示与历史合同，表的驻留位置不改变 Main 的消费语义。

磁盘路径从 V3 NF4 表按行读取，使用有界行缓存和准备/上传槽；输入已知部分可以增量准备，
计算消费稳定的 embedding 与就绪依赖。上传槽直到最后 GPU 消费者结束才复用。

双机 Device 路径按表 shards 放置，各 rank lookup 后合并精确 words，再执行复制的 PLE 计算。
数据生产可与前序计算重叠；Main 抓图不按磁盘或 Device 来源任意切碎。
PLE 缓存不属于请求 State，只有卷积历史和真实 token 前沿参与恢复。

实现入口：[PLE](../../src/models/qwen4_exp/ple/)、
[Device lookup 组合](../../src/models/qwen4_exp/program/device_ple.cpp)。

## 并行与放置

| 部分 | 双 Spark 分工 |
|---|---|
| GDN / Sparse GQA | 按完整 heads TP；O 产生局部贡献后归约 |
| QSA Indexer | 投影、评分、index cache/tail 复制，供本地 GQA 使用 |
| HC / residual | 复制；消费完整归约结果 |
| Routed experts | TP2/EP1 时每个 expert 的 intermediate TP；TP2/EP2 时完整 expert EP |
| Router / shared | Router 复制，shared intermediate TP；贡献统一汇合 |
| Embedding / head | Embedding 复制，head 按词表 TP |
| PLE / Vision | PLE 表分片；Vision 权重复制、独立编码单元分工 |

EP 保留全局 expert IDs 和 route 系数，不在本地集合重新归一化。TP 的 GU 与 down 沿对应
intermediate 同时切分，block-FP8 必要时保值 refine。共享专家只贡献一次完整数学结果。

Router/Indexer 的复制决策使用一致的输入与计算条件；模型需要共同的离散结果，不能用单个
Op 的近似精度容限替代一致性要求。具体切片、expert 映射与通信由
[ParallelPlan](../../src/models/qwen4_exp/parallel/plan.cpp)维护。

RTX 5090 还可通过[expert cache](expert-weight-cache.md)执行 Main experts 的 CPU/GPU 混合路径。
Spark 使用 GPU 计算；Context 每 rank 持有本地 State/KV，普通 TP histories 构成完整共同恢复点。

## MTP、Vision 与 Graph

MTP 使用私有 HC/QSA/MoE 层，与 Main 共享 embedding，默认共享 Main head；optimized 时只改变
其 draft head 绑定。Conditioning 为 Main 四流 residual，stem 保持这一特征关系。
稳定前沿、首步采用和真实 conditioning 的 history repair 见[Speculative 执行](speculative-decoding.md)。

Vision 复用 Qwen tower 与 MRoPE 输入机制。编码 features 与 prefill 窗口按真实媒体 spans 交接，
状态与前缀身份同时包含媒体和位置。资源机制见[Frontend](frontend.md)。

Resident decode 捕获连续设备主体，QSA 在图内更新本地 attention 节点；普通/MTP 的决定与最终
状态采用保留各自依赖边界。Offload 保留 CPU expert 所需交接，Prefill、Replay 与 Vision encoder
使用 eager。图目录和节点选择属于 [Program](../../src/models/qwen4_exp/program/)，
不在文档维护 executable 数量或逐 shape 的分派表。
