# Speculative 执行

Proposal、target verification、输出决定和状态采用共享[Engine 事务](engine-architecture.md#提交发布与退役)。
Backend 定义如何提议、需要什么历史和如何恢复；公共执行层不推导模型的前沿数学。

## 接受与正式采用

一轮先产生 drafts，再由 target 验证并形成 licensed 输出。Frontend 根据停止、EOS、预算和取消
选择最终前缀，Session 才提交对应输入状态。接受 draft 的数量、生成候选数量与最终采用输入数量
分别解释，不能在终态路径里直接互换。

Target 处理后续输入所需的 anchor 可以已经对外确认、但尚未执行。
采用 N 个输入后，State/history 和 continuation hidden 对应这一前缀；最后一个新输出成为下一轮
anchor。拒绝后缀不进入 checkpoint，相关 frame 在最后消费者结束后复用。

随机接受使用实际 proposal 分布 q 与处理后的 target 分布 p，拒绝校正使用相应残差分布。
Greedy 路线按 target argmax 接受。Grammar、penalty 和词表截断在各自数学位置生效，
不能裁剪 q 来伪造另一种 proposal。精确概率合同由
[sampling](../../include/ninfer/ops/sampling.h)与相应候选/acceptance Ops 维护。

## MTP 的稳定前沿

令 Main 已计算 `[0,F)`，已知下一输入为 `x[F]`。稳定状态为：

```text
Main frontier = F
MTP history frontier = max(F-1, 0)
saved conditioning = Main 在 F-1 的真实特征
```

MTP ordinal p 配对 Main 特征 `H[p]` 与 token `x[p+1]`；递归 draft 的后续步骤才使用预测层自身
hidden。Main conditioning 的具体表示由模型决定：Qwen3.5/GLM 使用最终 hidden，Qwen4Exp 使用
final HC mixer 前的四流 residual。媒体 embedding 的配对规则也归模型。

每轮从 canonical 状态执行 Draft → Verify → Decision → Commit。首步使用真实 conditioning，
满足配对与寿命条件时可直接采用它的 history；之后需要的列以本轮 Main hidden 重建。
Token 被接受不意味着递归 hidden 等于真实 hidden，因此不能只截短 draft cache 完成提交。

Prefill、Replay 与 Control 同样建立 backward-aligned history：根块跳过首 token 的配对，
后续块用上个块保存的 conditioning 衔接。History-only 路径只做建立 K/V 或 latent/index 所需的
stem、投影和状态更新，不执行无用的 query、attention、FFN 或 head。

临时 draft tail、selection 和 records 属于 frame；canonical history 与定长 tail 属于 Context。
首步候选、真实修补和零采用的规则在模型 Program 中维护，不按单机/双机另造状态语义。

## ReplaySSM

GDN/KDA 的 recurrent state 很大。短窗口 verify 保存每列完整 state 会放大容量与写流量，
因此保存一个原始 checkpoint 和驱动状态转移的紧凑 records，接受后只重放采用前缀。

```text
verify: S0 + inputs → outputs + transition records
commit(m): 从同一 S0 顺序重放 records[0:m] → Sm
```

重要的是有限精度转移一致，而非只在实数域代数等价。Records 保存同次 verify 实际消费的
raw key/value、gate 等 represented inputs；fold 使用同一 normalization、reduction、乘加和
state-store 过程。GDN scalar decay 与 KDA vector decay 保留各自形式，不合并 record 含义。

如果 fold 改写了求值次序，微小 state 差异会进入下一轮，不能以当前输出“看起来正常”证明正确。
精确验证比较同一物理输入块的 recurrence trajectory 与 fold 前缀，不能重新运行一个更短的
projection 充当参考。Chunked Prefill 则按自身数学精度验收，其末端 state 可以作为下一轮来源。

卷积 history 从旧尾部和实际输入 records 提取，保持 words。零采用不推进 State；
拒绝后缀不读取，源/目的与 records 保持到 fold 及最后消费者结束。
[具体 GDN](../../include/ninfer/ops/gdn_replay.h)和
[KDA](../../include/ninfer/ops/kda_replay.h)合同维护读写域与精确关系。

## DFlash 与 DFlash2

DFlash 从指定 target blocks 的完整 residual 特征形成 conditioning，再建立各 draft 层的 context
K/V。一轮输入为 anchor 加 masked columns，用非因果 masked-block attention 一次生成 proposals；
它不是逐步运行 MTP。Context prefill 只物化 conditioning K/V，不运行整套 draft FFN。

Local draft layers 使用 cyclic KV，full layers 使用 paged history。
Query block、候选路径和 proposal 概率属于本轮 scratch。DFlash2 增加动态卷积与条件候选选择，
其实际 q 来自候选路径，不能替换成各列独立 argmax 或 softmax。

DFlash proposal 是 one-hot；DFlash2 保留实际条件分布。它们在请求尾部的物理 proposal 宽度与
有效 target 宽度可以不同；缩短非因果 block 可能改变全部 proposals，不要求前缀结果相同。

Target 提交后，已采用的 conditioning 可以暂存为待追加特征。下一次 proposal 或保留完整
checkpoint 前补齐 draft context；临时特征本身不是完整 checkpoint。恢复需要 target State、
draft context 与各自 frontier 同时一致。

组合代码见 [Qwen3.5 draft](../../src/models/qwen3_5/execution/draft.cpp)，
数学接口在相应 attention、动态卷积和候选选择 Ops 中。

## Graph 与生命周期

图保持稳定输入、frame 和状态寻址，按必要的计算容量组织，实际有效列、accepted 长度和采样
参数作为设备输入。Host grammar 依赖 draft 时，先读回候选并与 target forward 重叠，
decision 等待 mask；最终提交仍由真实采用前缀驱动。

Record/fold、MTP history repair、tail 和 hidden 更新可以进入提交图，但图不拥有逻辑 frontier。
取消结束采用意图，设备消费者继续按 ticket 退役。完整工作许可在 verify 前覆盖后续提交峰值，
不能在 acceptance 已产生之后才向公共池申请必需空间。
