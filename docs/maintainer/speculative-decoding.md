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

## 逐轮验证长度与容量授予

`draft_tokens` 在启动时确定窗口上限 K，最大 target 宽度为 `W_max = K + 1`。
若 Main 前沿为 F，Decode 准入只预留推进到 `F + 1` 的 anchor 及所需的 backend state 和 COW。
额外 draft 增长由 Session 在上下文和输出预算内向 `ContextDirectory` 请求即时可用容量。
可选授予不驱逐缓存前缀、不触发抢占。每行验证深度 `v_i` 受 `granted_limit` 约束。

`SpeculativeLengthPolicy` 的 Fixed 与 Adaptive 共用 `DecodeRoundPlan`、frame 视图和提交路径。
启用 speculative backend 后，每个结算的真实 Decode 行步恰归入一类：
验证轮 `v_i > 0`，策略旁路 `v_i = 0` 且 `granted_limit > 0`，或边界/容量回退
`v_i = 0` 且 `granted_limit = 0`。物理图宽可以大于当前有效列数。

## 置信度读取、同轮收窄与无偏性

跨轮控制（Tier C）按序列维护指数衰减的逐位置到达与接受计数，用分层收缩估计条件存活率。
`W = 1` 普通 Decode 的 target 全支持集 top-1 概率 `p_top1` 为旁路后的下一轮提供唤醒信号，
低熵转变可直接进入估计，省去周期性强制探测。TP 词表分片通过 AllGather 汇合各 shard 的
FP32 最大分数与相对该最大值的 log-sum-exp，得到全词表域的 `p_top1` 和 MTP draft head 的
top-1 置信度，精度遵循[词表 Op](../../include/ninfer/ops/vocabulary.h)合同。MTP 实际 proposal 为
argmax 的 one-hot 分布，DFlash2 的 `q(d)` 由候选路径条件分布给出。

Qwen3.5 MoE 的 Adaptive `SplitProposal` 路径（如 CPU expert offload）可在同轮 target forward
前按抽样前的置信度收窄（Tier S）。DFlash 读取 head top-1 概率，DFlash2 按已采样前缀逐位置
使用停止条件概率，MTP 读取抽样前 top-1 概率。收窄满足 `W_exec <= W_init` 和
`proposed_tokens >= drafted_tokens`。DFlash2 按前缀顺序决定每列是否进入验证，再选择覆盖该深度
的物理路由。随机 proposal 的 `q != p` 时，不以已抽中 token 的 `q(d)` 决定是否验证该列；
验证仍使用实际条件 proposal 分布。

## 物理成本模型、启动校准与窄图门控

`SpeculativeCostModel` 以 `sum_i L_i(v_i) / cost_ns` 评价候选，`L_i` 为该行期望产出 token 数。
成本特征覆盖 target token、按上下文加权的 attention、MoE 唯一 expert 数的非线性饱和、
MTP 步数与行数以及 DFlash block/head 工作。Adaptive 启动时以
`StartupPhase::SpeculativeCalibration` 在临时 Context 上测量代表性的边界几何，使用 COW 分支
保全共享前缀，并拟合非负权重。在线按 `DeviceRetired` 的总延迟，以实测/预测比的 EWMA 修正成本。

Adaptive masked-block 路线的窄 target 图（`1 < W < K + 1`）按扣除 Graph 资源后可保留的
Device Context 容量门控。自动容量下，窄图方案须保留至少 95% 的全宽基线容量，该比较同样
适用于 NVFP4、K8V4 等块量化 KV。SplitProposal 的 `ProposalGraph` 在同一物理 batch 桶内按
proposal 几何复用，多个 target 宽度可共享同一 masked-block proposal 图。

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
后续块用上个块保存的 conditioning 衔接。多请求打包 Prefill 通过 ragged `prepare_causal_pairs`
在单次调用内跨异长段构造对齐配对（仅对 `main_frontier == 0` 的段跳过首列，对续算段衔接源 State 保存的 conditioning），
再通过原生 ragged KV/prepare 入口一次性写入历史。History-only 路径只做建立 K/V 或 latent/index 所需的
stem、投影和状态更新，不执行无用的 query、attention、FFN 或 head。

临时 draft tail、selection 和 records 属于 frame；canonical history 与定长 tail 属于 Context。
首步候选、真实修补和零采用的规则在模型 Program 中维护，不按单机/双机另造状态语义。

## ReplaySSM

GDN/KDA 的 verify 从只读的初始 recurrent state $S_0$ 出发，产生 BF16 readout 和紧凑的私有转移
records。$S_0$ 与折叠结果 $S_m$ 的存储格式遵循 `linear_state_dtype`（默认 FP32，可选 BF16，写回时单次
RNE 舍入），内部递推累加维持 FP32。每列保存表示后的 raw key、求解出的 FP32 修正 $\delta_t$ 和 FP32 衰减 $\alpha_t$。
GDN 使用每个 value head 的标量衰减，KDA 使用每个 key channel 的衰减。由于修正由 verify
求解，records 只可从当时观察到的同一 $S_0$ 折叠。

令 $\hat{k}_t$ 为归一化 key，令 $D_{a:b}$ 为从列 $a$ 到 $b$（含）的衰减逐元素积，空积为 1。
采用前 $m$ 列时，recurrent state 的闭式为：

$$
S_m = S_0 \operatorname{diag}(D_{0:m-1})
      + \sum_{t=0}^{m-1} \delta_t
        \bigl(\hat{k}_t \odot D_{t+1:m-1}\bigr)^\top.
$$

Fold 可分块流式计算该式。Readout 与 fold state 均按表示后的输入逼近数学递推，以独立数值参考
验证，并检查连续多轮采用后的轨迹。投影、因果卷积及激活也按各自的数学合同验收。

卷积 history 按 BF16 words 精确取 `tail_3(source.conv || conv_record[0:m])`。Verify 的无效输出
列写入 BF16 `+0`，transition record 的无效后缀和源 State 保持原样；卷积 record 只定义有效前缀。
零长度 verify 不读 State selector 或字段，零采用 fold 不访问 State 或 records。Fold 仅读取采用
前缀，records 和元数据保持原样。帧继续持有 records，State 保留到最后一个消费者完成。具体读写域见
[GDN](../../include/ninfer/ops/gdn_replay.h)与[KDA](../../include/ninfer/ops/kda_replay.h)合同。

## DFlash 与 DFlash2

DFlash 从指定 target blocks 的完整 residual 特征形成 conditioning，再建立各 draft 层的 context
K/V。一轮输入为 anchor 加 masked columns，用非因果 masked-block attention 一次生成 proposals；
它不是逐步运行 MTP。Context prefill 只物化 conditioning K/V，不运行整套 draft FFN；多请求打包 Prefill
按各段窗口尾部截断后，通过原生 ragged 路径一次性物化所有段的局部 cyclic 与全长 paged 上下文。

Local draft layers 使用 cyclic KV，full layers 使用 paged history。
Query block、候选路径和 proposal 概率属于本轮 scratch。DFlash2 增加动态卷积与条件候选选择，
其实际 q 来自候选路径，不能替换成各列独立 argmax 或 softmax。

DFlash proposal 是 one-hot；DFlash2 保留实际条件分布。它们在请求尾部的物理 proposal 宽度与
有效 target 宽度可以不同；缩短非因果 block 可能改变全部 proposals，不要求前缀结果相同。

Target 提交后，已采用的 conditioning 可以暂存为待追加特征。下一次 proposal 或保留完整
checkpoint 前补齐 draft context；临时特征本身不是完整 checkpoint。恢复需要 target State、
draft context 与各自 frontier 同时一致。

组合代码见 [Qwen3.5 draft](../../src/models/qwen3_5/execution/draft.cpp)与
[GLM5Next DFlash2](../../src/models/glm5next/execution/dflash.cpp)，
数学接口在相应 attention、动态卷积和候选选择 Ops 中。

## Graph 与生命周期

图保持稳定输入、frame 和状态寻址，按必要的计算容量组织，实际有效列、accepted 长度和采样
参数作为设备输入。Host grammar 依赖 draft 时，先读回候选并与 target forward 重叠，
decision 等待 mask；最终提交仍由真实采用前缀驱动。

Record/fold、MTP history repair、tail 和 hidden 更新可以进入提交图，但图不拥有逻辑 frontier。
取消结束采用意图，设备消费者继续按 ticket 退役。完整工作许可在 verify 前覆盖后续提交峰值，
不能在 acceptance 已产生之后才向公共池申请必需空间。
