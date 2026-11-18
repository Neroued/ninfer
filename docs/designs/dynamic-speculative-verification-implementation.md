# 动态投机解码实施计划

[设计文档](dynamic-speculative-verification.md)定义最终的轮次、数值和资源合同。本文供实施者使用，说明每一步要切换哪些生产路径、交给下一步什么，以及用什么证据判断已经完成。起点为 `dev@ebdc67e5`。

## 工作顺序

| 步骤 | 交付的能力 | 依赖 |
|---|---|---|
| S1　轮次计划与执行协议 | 给定成员与逐行长度，可以唯一解释生成、target、decision 和 commit 的工作。协议可传给两个 rank。 | 现有模型准备与 Work |
| S2　必需许可与可选增长 | Context 紧张时，能推进的请求保持可执行。最终计划有足够的空间完成整个事务。 | S1 的逻辑长度与身份 |
| S3　设备视图与算子 | 矩形短窗和紧凑异构窗能计算同一套数学，并写入可供原生提交使用的 records。 | S1 的布局合同，可与 S2 并行 |
| S4　三套模型的 eager 全链路 | 通过实际模型执行指定长度，覆盖普通、MTP、DFlash、DFlash2、提交和恢复。 | S2、S3 |
| S5　Graph、分布式与成本测量 | 正式设备路径按真实几何减少计算，两个 rank 对同一计划执行同一通信序列，得到可用成本曲线。 | S4 的同源 eager 层组合 |
| S6　自动选择与产品验收 | 在线策略、配置、统计与 bench 共用正式执行路径，按实际负载确认收益及代价。 | 策略数学可在 S1 后独立开发，产品启用依赖 S5 |

在同一迁移分支直接替换内部合同。步骤边界可以只完成 Host 合同、一个算子族或一种模型的执行。尚未切换的调用方可以暂时无法构建或运行。依赖模块直接迁到目标接口，两台 worker 使用同一版协议。`fixed` 是最终产品的固定长度策略，与 `adaptive` 共用模型和提交实现。完整构建、真实服务和产品可用性在整合边界验收。

## S1　冻结一次 round，贯穿消息与 frame

**要解决的问题。** 当前三套 Session 已经能按剩余预算裁掉行尾，但 Program 多处从启动配置重建实际宽度。只有当 records、目标 logits、结果步长与 Graph 都消费同一计划时，短窗才会产生真实收益。

### 轮次对象和目录

在 `src/models/common/speculative/` 增加共享的计划值与几何构造。建议由 `round.h/.cpp` 承担计划、物理布局与确定性映射，`policy.h/.cpp` 承担 S6 的估计与选择。已有 [BatchExecutionPlan](../../src/models/common/batch.h) 继续负责行容量。模型的权重、前沿、history 数学和 Op 选择仍在各自模型实现中。

以下是数据边界。名称描述要交接的事实，派生数组由 builder 一次构造，传输时只编码恢复它所需的事实。

| 值 | 最少需要保存的内容 |
|---|---|
| 原生 Work 的成员 | 逻辑请求顺序、sequence generation、residency、frontier、anchor、已有模型 selectors/owner/sampling。当前批次已拥有的内容直接复用。 |
| `DecodeRoundPlan` | 逻辑顺序的 `verify_drafts[B]`、选定的 target 几何，以及选定 proposal 图段的物理几何。Ordinary、矩形和紧凑采用带类型的表示。 |
| Target 几何 | 矩形使用已有的全局/本地行桶和 `width`。紧凑使用全局 token 桶及需要时的本地 token 桶。初始化时能从这些事实恢复 frame 的实际视图。 |
| Proposal 几何 | MTP 保存实际采用的每步物理形状，步数由长度向量的最大值确定。DFlash 由已加载模型给出 block 宽度，保存 proposal 活跃行桶与 head 的物理列桶。全零时为空。 |
| Frame 内派生表 | owner 稳定行序及逆映射、segments、有效 token 数、各步活跃行和前一步结果位置、record 行。表在每次提交时重建一次，覆盖此次可能读取的物理范围。 |

共享部分的目标接口可以收敛为下面的值类型。`MtpSteps` 的元素描述当步活跃成员所选的物理桶，`MaskedBlock` 描述 block 和 head 的工作几何。普通、矩形和紧凑几何各自携带该表示的必要字段。

```cpp
struct DecodeRoundPlan {
    std::vector<std::uint32_t> verify_drafts; // indexed by logical member
    std::variant<NoProposal, MtpSteps, MaskedBlock> proposal;
    std::variant<OrdinaryTarget, RectangularTarget, PackedTarget> target;
};
```

Session 将此值与成员身份装入原生 Work，完成目录许可后提交。RankProgram 根据已选几何在本地 frame 建表，传给 Op 的是 Tensor、行边界和必要的状态 selectors。模型层可以重排物理行，原始 `verify_drafts` 始终按逻辑成员解释。

修改三套模型的 `work.h/.cpp` 与 Session 事务。Qwen3.5 将 `DecodeRow::draft_extent` 的权威值收进 round。Qwen4Exp 和 GLM5Next 将现有 `draft_extents` 与物理执行描述归入同一计划。Qwen3.5 可继续用 `DecodeWork` 表达 ordinary，另两套模型可继续使用 `ForwardKind::Ordinary`。在现有 native variant 中表达同一语义即可。[Qwen3.5 Work](../../src/models/qwen3_5/work.h) · [Qwen4Exp Work](../../src/models/qwen4_exp/work.h) · [GLM Work](../../src/models/glm5next/work.h)

Qwen4Exp/GLM 的模型 `wire.cpp` 编解码 Work。其 `parallel/wire.cpp` 传启动/分片准备事实，`parallel/codec.cpp` 和 `WorkContract::payload_capacity` 界定实际消息上限。逐一切换。rank 取得同一份长度、几何和逻辑身份，按约定顺序还原映射。只有 rank-local 的资源绑定因 rank 而异。上限还需覆盖最大成员数、每行合法候选、约束行的 `v+1` 个 mask 和 Result/Commit。一次 mask 大小由实际词表决定。沿用当前有界传输设施，避免在 Work 中重复发送可导出的逐 token 表。[公共启动 wire](../../src/models/common/parallel_wire.cpp) · [Qwen4Exp Work wire](../../src/models/qwen4_exp/wire.cpp) · [GLM Work wire](../../src/models/glm5next/wire.cpp)

### 一次事务在各边界交接什么

| 方向 | 在现有消息上建立的约束 |
|---|---|
| Session → RankProgram，Work | 精确的逻辑行、送验数和物理计划。Session 保证已获额度，Work validator 验证行映射、几何、值域及该模型准备的目录。 |
| RankProgram → Frontend，DecisionRequest | 只列约束成员，行号指向冻结的逻辑顺序。每行附上 `vᵢ` 个候选。普通轮及 `vᵢ=0` 的约束行仍需要当前位置的 mask。 |
| Frontend → RankProgram，DecisionInput | 每行 `1+vᵢ` 个 mask、对应词表 word 数、dead-position 位图及同一 stage 标识。发送方与 staging 按 DecisionConsumed 边界交接。只有实际经过的位置触及 dead end 才结算失败。 |
| 结果 owner → Session，ResultReady | 正常 decode 行 `processed=1+vᵢ`，报告实际提案事实、`aᵢ` 和 `aᵢ+1` 个 licensed IDs。送验数必须与 Work 精确相符。多 rank 返回一次逻辑结果。 |
| Frontend preview → Session → RankProgram，Commit | 用逻辑行提交 `cᵢ` 及 terminal/cancelled/failed。依 Work 的映射在每个 rank 找回原 records，检查 `cᵢ` 在 licensed 前缀内，再交付执行。 |
| 全部参与 rank → Executor，DeviceRetired | 只结算对应 ticket/command 的租约与批次计时。部分 rank 失败按现有失败协议停止提交并排空，保留原始错误。 |

共享计划代码消费成员顺序、长度和几何。各模型的 MTP 首配对、PLE 输入、KDA selection 等留在模型自己的事务和 frame 中。[Executor 合同](../../src/runtime/executor/contract.h) · [现有 Frontend mask 交接](../../src/runtime/engine/engine_execution.inl)

**交付判断。** DFlash2 用 `[7,3,1,0]`，MTP 用 `[3,1,0]`，另构造全零和 owner 顺序打乱的 Work。一次本地解释及一次模型 wire round-trip 后，逐行 `processed=1+v`、总 target 位置、物理容量、decision 成员与返回顺序都能精确列出。真实成员缺 anchor、长度越出已授予范围、桶装不下或跨成员关联错位，由所属合同拒绝。测试和开发 bench 通过同一 builder 指定合法计划，后续步骤可以独立验证执行。

## S2　把 Context 许可从窗口上限中拆出来

**入口。** 现有 `reserve_resident_unit` 先逐请求取得 unit，`build_round_membership` 随后选批。Qwen4Exp/GLM 的 `unit_end` 和 Qwen3.5 的 `next_unit` 把 `K+1` 纳入前置授予。这里需要改变许可覆盖范围，保留 Scheduler 的公平轮转与恢复保护。[Engine 调度](../../src/runtime/engine/engine_schedule.inl) · [Qwen3.5 需求](../../src/models/qwen3_5/session/request.cpp) · [Qwen4Exp 预留](../../src/models/qwen4_exp/session/context.cpp) · [GLM 预留](../../src/models/glm5next/session/execution.cpp)

1. **取得基本工作许可。** Decode 的必需前沿为 `F+1`，带上模型派生的 Main/backend 写入、源/目的保护、共享尾页 COW、已获恢复保留。Runtime 继续提供 `RoundBudget`，必需许可仅由原来的请求 unit 及其 generation/residency/frontier 失效条件决定。Prefill、Replay、Control 各按自己的完整需求授予。
2. **选批并申请可选页。** Session 收齐成员后求偏好长度，先取得 Executor frame。用各 history 的 `write_pages`、已有 growth 及其 metadata demand 构造联合申请。只接受可立即交付的容量。ContextDirectory 应提供复用现有布局与可行性计算的即时授予入口，调用者能区分普通短缺与目录/设备错误。可选分支跳过需要维护、等待或回收才可取得的额度。
3. **压力下缩短。** 整批首选方案不能立即授予时，先保留各行已覆盖的合法前缀。把需要新资源的后缀按页/COW 需求发生变化的位置分段。每段是该行所有相关 rank 和 history 的原子申请。优先尝试预期收益较大的可达段，平局沿用选批顺序。一行某段失败，其他行仍可使用自己的 owner 容量。有界扫描只访问该批的增长断点，正常路径一次联合申请，避免每个 token 重扫目录或候选缓存。
4. **冻结和结算。** 在可授予范围内确定最终计划并归还未用的临时额度，再 `prepare_write`、准备 actions、submit。frame 不可用、取消或提交前失败时撤销尚未采用的可选预留。已经发出的映射/COW 进入现有事务退役。commit/settle 以正式覆盖及 `recovery_peak` 为准。必需申请本身失败才回到已有 reclaim/暂停路径。各模型的 bind、release_units、pause/resume、checkpoint 和恢复后第一次新增进展要使用相同的必需前沿。[目录与事务](../../src/models/common/context/directory.cpp) · [History 结算](../../src/models/common/context/directory/history.cpp)

基本许可在等 frame 时仍属于原请求。可选增量在 Session 所有的待提交事务中保存，先保证整批各行的必需进展，再分配剩余资源。对于没有新增页或 COW 的普通轮，直接复用已授予的范围即可。`unit_runnable` 的等价检查继续绑定到原请求和原 unit，重新报价可选列不会使已经就绪的 anchor 失效。

**交付判断。** 覆盖私有页尾 `F=63` 且无新页、同一位置共享尾页必须 COW、GLM 仅一个 owner 有空间、恢复保留与其他请求争用四种情形。明确断言能否提交 anchor、各行最终范围、是否发生暂停/回收以及退役后可归还的额度。共享尾页的已保留前缀应能继续恢复。

## S3　固定记录地址，让算子读取真实的 token 段

先让矩形路径支持 `W<Wmax,B>1`。这一步可以独立暴露 stride 错误。随后为同一数学接口接入紧凑布局。建议让消费多请求序列的 Op 共用一个只读 token 段视图，给出物理 token 容量、行容量和 Device I32 边界数组。需要 token→行或行→record 映射的算子借用 frame 已有表。连续的逐列单元直接使用 Tensor 原生 token 轴。

**Frame 与持久记录。** Qwen3.5 的 `ExecutionFrame/decode_view`、Qwen4Exp 的 buffers、GLM 的 `reshape_frame_controls` 目前都带有启动窗口的假设。为当轮计划分别绑定 proposal、target、decision 和 commit 视图。对 logits/hidden/feature 每个 producer 写出后，保留其地址、offset 或 stride 给最后消费者。每个真实行的 licensed IDs 写入稳定的输出行槽，长度单独交给 Result/Commit。只传输合法数据及合同要求的控制信息。DFlash 候选及 `q` 的步长随 proposal 输出视图一起移交。Host 读回、grammar mask、计数更新和 `last_columns` 共同使用同一套行标识。[Qwen3.5 帧](../../src/models/qwen3_5/program/frames.cpp) · [Qwen4Exp 缓冲](../../src/models/qwen4_exp/program/buffers.cpp) · [GLM 帧](../../src/models/glm5next/program/frame.cpp)

| 算子或调用链 | 需要实际完成的改变 | 验收重点 |
|---|---|---|
| verify inputs、位置、RoPE、embedding、causal pairs | 按逻辑行写入 anchor 与 draft 前缀，构造段内 ordinal 和多轴位置。历史配对使用本行前一 hidden/坐标。矩形快路沿用连续寻址。 | 连续两行的边界、首列与 bonus 位置准确。空输入不读隔壁行。 |
| GDN/KDA 输入投影、卷积、recurrent record、fold | 分离输入 token 布局与记录 row/layer stride。矩形短窗、紧凑行各自寻址，records 总按稳定槽写入。KDA 的投影输出 K/V 和后续 recurrence 读取支持同址。fold 按已保存的行与采用列重放。 | `Wmax>W`、部分行一列、哨兵尾部、行重新排列。独立数学正确，fold 与同一物理输入产生的前缀状态逐位一致。 |
| Dense paged GQA、QSA 和 Kpool MLA | 让 paged query、KV 写入、index prepare、selection、稀疏 readout 获取正确的段与页表。GLM 对 owner-local 段工作，并能复用 MTP 首步的冻结 selection。既有量化 KV codec、cache 读写和上下文 profile 属于各 Op。 | 长短历史混合、4-token 压缩边界、64-token 页边界、空 owner、selection 复用。检查选块身份、tail 和因果可见域。 |
| PLE | lookup/hash 和卷积以请求划界，records 按 ticket 行存放。Host/SSD producer 接收实际需读的 anchor/前缀。Device 视图能直接对应本轮 token 布局。 | 有效行数、前两个原始 ID、零 draft 行、丢弃后缀及异步窗口的最后消费者。 |
| Linear、HC/mHC、MoE 和 TP 输出 | 将矩形 `P×W` 或紧凑 `Q` 作为矩阵的 `M`。更新 MoE 有效域，紧凑布局只屏蔽末尾填充。核对 fused、expert TP/EP 和 CPU offload 的 assignment、combine、容量查询与 collective。复用已调优的 contraction。 | 同一有效输入通过真实入口得到正确输出。padding 无 expert 访问，区分有效 token 与决定通信/模板的物理 token。 |
| Draft head、selector、单机/分片词表采样 | `rmsnorm_pack_tail` 或 DFlash v1 的提取逻辑按行输出所需列。局部 top-k 与全局合并使用同一候选列序。selector 的前驱定位和 RNG 使用本行坐标。accept 支持 prefix 对应的实际 logits、candidate IDs、`q` 及其 stride。 | 例如 `B=4` 时裁剪使 Spark MXFP8 head 从 28 列变成 12 列，跨越 A8/A16 分派。核对实际稀疏 `q`、零 draft 行、全接受、首拒、混合采样、penalty 和 mask。 |

对应入口见 [sequence Ops](../../include/ninfer/ops/sequence.h)、[GDN records](../../src/core/gdn_replay_records.cpp)、[KDA records](../../src/core/kda_replay_records.cpp)、[QSA](../../include/ninfer/ops/qsa.h)、[Kpool MLA](../../include/ninfer/ops/kpool_mla.h)、[PLE](../../include/ninfer/ops/ple.h)、[MoE 有效域](../../include/ninfer/ops/moe.h)与[分片 vocabulary](../../include/ninfer/ops/vocabulary.h)。实现放回原 family，并更新对应 target 的 source manifest 与 workspace 上界。现有 `prepare_ragged_prefix` 输出仍是 `[D,W,B]`，只能在其合同行距内复用。

覆盖各入口已有的表示。Qwen dense paged GQA 与 QSA 保留 BF16、INT8、FP8、NVFP4 和 K8V4 的读写及 scale 平面寻址。GLM 保留 BF16/FP8 latent 和该部署选择的 indexer 表示。GDN/KDA 的量化投影覆盖现有 recipe，包括 GLM FP4 Local。优先在共享的行列寻址处接入，只有依赖连续跨度的私有 kernel 才需要单独修改。Vision 用的 packed attention 与 decode 用的 paged causal attention 有不同的 KV 和因果合同。[Dense attention](../../include/ninfer/ops/softmax_attention.h) · [GLM 表示与部署](../maintainer/glm5next-model.md)

普通 Tensor 继续描述连续存储或已有 strided view，行边界只交给需要请求归属的计算。MoE 是可以直接复用的例子。对已压紧的全局输入，现有 `MoeTokenValidity` 可取 `tokens_per_row=Q`、`valid_columns=[N]` 表示一个连续有效前缀。这个行宽只用于路由有效性。offload 调度仍接收真实活跃数 `N`。[MoE operands](../../src/ops/moe/operands.cpp)

**交付判断。** 用同一组成员验证矩形和紧凑布局的逻辑输出、记录写域、源保护与 padding。判断基于独立参考和 Op 公开精度。布局精确操作比对原始 words，浮点 contraction 按原 policy 评估。优先扩展[已有序列](../../tests/ops/sequence)、[接受](../../tests/ops/sampling)、[SSM](../../tests/ops/linear_attention)和[注意力](../../tests/ops/softmax_attention)用例，每个 family 保留捕获具体错误所需的 case。

## S4　逐模型打通 eager 的生成、验证和提交

每个组合先以指定的 `v[]` 经过真实 Session、Executor、RankProgram、Decision 与 Commit。单 rank 走 LocalExecutor，多 rank 走原有控制和通信路径。实际 `Work` 完整运行后再测策略，避免执行宽度与预测质量混在一次排查中。

### GLM5Next

首先接 DFlash2。现有 `prepare_speculative`、`dflash_run`、`accept_dflash` 及返回 ID 的行距都直接读取 `shape.draft_tokens`。把完整 block、head 候选列、target logits 范围和 commit 源视图拆开。采用 `v=5` 时必须能观测到完整 7-draft block、每行五个有效候选、六个 target 位置。GLM 的 DFlash2 采用后追加 cyclic KV，按真实 `c` 聚合各层 context 物化工作。[Program](../../src/models/glm5next/program/speculative.cpp) · [DFlash2 执行](../../src/models/glm5next/execution/dflash.cpp) · [提交](../../src/models/glm5next/program/commit.cpp)

随后接 MTP。`mtp_forward` 的递归 hidden、raw tail scratch、selection 和可见上界目前按固定的局部/全局行寻址。第一步保存可采用的 history 和 selection。后续活跃集合收缩时，按原行找到前一 hidden、已选 pool 及 frozen end。保留 owner-local MLA、全局 KDA 和 FFN 的映射区别，双 rank 上空 owner 也参与所需 collective。零 draft 行的 history-only 配对和其余行一起提交。[MTP Program](../../src/models/glm5next/program/mtp.cpp) · [物理输入](../../src/models/glm5next/program/inputs.cpp)

### Qwen4Exp

将 `prepare_mtp` 中固定次数的循环、`draft_visible`、`step_inputs`、词表读回和 `prepare_main` 改为使用冻结计划。为需要生成的行执行 MTP。模型在提交时用四流 conditioning 和真实 RoPE 对齐配对。`rebuild_history` 的判定要包含首个真实配对是否已产生，单靠 `c>1` 无法覆盖所有混合行。QSA tail、PLE 卷积和 GDN fold 共用稳定 record 行。[MTP](../../src/models/qwen4_exp/program/mtp.cpp) · [提交](../../src/models/qwen4_exp/program/commit.cpp)

单机 Host/SSD PLE 随 anchor 及已产生的有效 draft 提前开始读，`append_device` 的行映射随每步活跃集合传递，`seal` 以最终各行输入数为界。窗口忙时沿用 ticket 的等待与重试，取消后等实际生产/消费退役。Device PLE 路径直接消费同一数据，不为宽度选择增加一次跨机消息。检查 expert offload 用的是实际活跃 token 数，Graph 和 workspace 规划用的是可执行几何。[PLE pipeline](../../src/models/qwen4_exp/program/ple_pipeline.h) · [MoE 调度](../../src/models/common/moe/schedule.h)

### Qwen3.5 系列

在 `initialize_decode` 后只从 Work 取宽度与布局。MTP 的首步和递归步分开处理，Qwen 的真实 conditioning、history skip 和本轮采用前缀共同决定修补范围。DFlash v1 使用独立 proposal 可见长度。DFlash2 从完整 block 的输出取前缀并保留真实条件 `q`。同时覆盖 full 和该权重已有的 optimized head，涉及精简词表时保留全局 ID 映射。[Decode](../../src/models/qwen3_5/program/decode.cpp) · [DFlash 执行](../../src/models/qwen3_5/execution/draft.cpp)

重点追踪 `pending_features` 在连续 ordinary、下一轮 proposal、终止保留、pause 与 restore 时的寿命。记录待追加区间的起止位置和 storage owner，下一次写入该槽前完成追加。Session 的 continuation、paged/cyclic context 与 normalize 完成事实一致时才可保存相应 checkpoint。[DFlash 提交](../../src/models/qwen3_5/program/draft_commit.cpp) · [暂停路径](../../src/models/qwen3_5/session/pause.cpp)

**交付判断。** 对每种后端在同一请求运行“长窗 → ordinary → ordinary → 短窗 → 长窗”，再覆盖一轮长短混合、提前 stop 和中途取消。核对 Main/backend 前沿、最后采用的输入、tail、counts、checkpoint 后的首次续算。GLM 实际模型在双 Spark 验证，Qwen3.5 家族使用 5090，Flash Next 覆盖单 Spark 的 Host PLE 与双 Spark 的 Device PLE。可复用 [GLM Session fixture](../../tests/models/glm5next/test_session.cpp)、[Qwen3.5 MTP history](../../tests/models/qwen3_5/test_mtp_history_real.cpp)、[DFlash2 真实模型](../../tests/models/qwen3_5/test_engine_dflash2_real.cpp)和[分布式执行测试](../../tests/distributed/test_execution.cpp)。

## S5　按物理工作构建 Graph 并测量成本

1. **由同一份 prepare plan 枚举目录。** 每个 backend 列出 ordinary、生成、verify、decision 和 commit 的可达几何。紧凑 target 和 GLM 本地桶遵守设计文档的上界。相同物理计算共用 executable。固定配置只准备其可达路线，预算尾部也有合法执行方式。按 rank 事先点出各类图的数量。
2. **保持模型计算段完整。** Qwen3.5 的 `TextGraph` 沿用 CPU expert 等实际交接。Qwen4Exp 的 Host PLE 用 producer handoff，Device PLE 可保留常见满窗整轮图。GLM 将 DFlash block 从 target 的 ADP `L` 维拆出，MTP 的首步与递归图消费预上传的逐步行表。rank 根据同一 Work 排好所有 launch，生成与 target 之间通过设备依赖衔接。Grammar mask 仍在明确的 decision 边界交付。[Qwen3.5 图](../../src/models/qwen3_5/program/graphs.cpp) · [Qwen4Exp 设备图](../../src/models/qwen4_exp/program/device_graphs.cpp) · [GLM 图](../../src/models/glm5next/program/graphs.cpp)
3. **保留局部更新与通信契约。** QSA/Kpool/滑窗 attention 按本地可见范围更新相容的 child。带 collective 的段整体选择与记账，每个参与 rank 的顺序、元素数、通信后端与 graph replay 通知一致。commit 的本地修补可以按本地采用数跳过。涉及 collective 时使用由全批提交导出的共同几何。[图通信合同](../../src/models/common/graph.cpp)
4. **重算真实高水位。** 检查 `P/L/W`、全局/本地 token 桶、不同 MTP 活跃集合、head、history-only 与提交临时量。frame 共用 backing，跨阶段存活量按重叠计费。workspace 覆盖非单调分派，含 offload 与词表 merge。将 transport、pinned staging、Graph 驱动开销与 Context 分开。用正式启动的 warmup/capture/rebuild 校准，核对每 rank 显存、Context token capacity、启动时间，和 GPU timing 关闭时的可执行性。[Graph 校准](../../src/models/common/graph_preparation.cpp) · [GLM 启动规划](../../src/models/glm5next/program/planning.cpp)

### 用指定长度测成本

| 样本 | 保持一致的条件 | 要回答的问题 |
|---|---|---|
| GLM DFlash2，双 Spark，C1 | 同一 FP4 Local 实例、full head、8 个 block 位置，分别验 `v=3/5/7`。采用与历史代表负载相近的上下文和采样。 | 少算 2 或 4 列时，完整 step 节省多少？按本次满窗的接受量与耗时重算盈亏线，同时列历史 83.42 ms 参照。 |
| GLM DFlash2，C8 | 同一语料与 owner 放置，分别比较等长及两组 `[7,3,1,0]` 的矩形、紧凑路径。再覆盖一例 ADP 偏斜。 | 相同逻辑前缀下，紧排省下的计算、通信能否抵消映射和图段发射？ |
| MTP，C1/C8 | 同一已加载的 backend 与 head，比较 `v=0/1/2/3`，并覆盖上限 5 时只有少数行需要深层的情形。 | 是否真的少跑 MTP 步？多一行和多一个递归深度各付出多少？ |
| Qwen 设备路径与大并发 | 5090 原生 Qwen3.5 系列、单/双 Spark Flash Next，分别采用其正常部署。已使用 16/32 并发的部署增加相应观测点。 | 固定宽度、Host PLE、CPU expert 或通信交接是否抵消小 batch 收益，目录成本是否随并发合理增长？ |

先记录物理 target/head 列数、MTP forward 次数和 collective 字节，确认节省发生在哪个阶段，再读 step ms。Op/模型 bench 用来验证局部工作量。端到端 forced plan 由开发测量构建在 Session 的计划入口指定，沿生产 Engine 提交同类请求，沿用已有 bench 的实际 batch 和完整退役计时。正式 Graph 与 eager 各自建立成本曲线。`v=0` 的比较包含已加载 backend 的 history 维护。[引擎 bench](../../bench/inference/ninfer_bench.cpp) · [现有模型 bench](../../bench/models/qwen3_5)

## S6　接入策略、观测和产品路径

**策略。** 在共享模块中实现设计文档的逐位置到达/通过计数、收缩先验、带预算的探测和真实计划评分。各模型 Session 复用该逻辑，并提供自己的可达几何和成本。按 ticket 在可提交结果处保存当前请求的质量标签，commit 接受后更新一次。`DeviceRetired` 提供一次完整批次成本。暂停对象携带学习状态，公共 prefix、设备 records 和 rank wire 都无需保存预测计数。Host 可以用指定计划的样本按时间顺序回放决策，检验冷启动、深位缺样、`v=0` 后重新探索和请求内容切换。评估时只向算法提供该轮之前可获得的信息。

**配置和指标。** 在 [SpeculativeOptions](../../include/ninfer/types.h)、[公共配置](../../src/product/config)、[启动 wire](../../src/models/common/parallel_wire.cpp) 和启动日志接入 `length_policy`。补齐 [示例配置](../../ninfer.example.toml)、相关用户文档及 CLI/serve/bench 的共享解析。单实例实际加载哪些 spec 权重仍由模型配置决定。示例只有在对应部署有收益后才改为 adaptive。

| 统计与所有者 | 计算口径及消费者 |
|---|---|
| Session 的行结果 | 逐行区分候选生成 `g`、送验 `v`、target 接受 `a`、采用的 draft `min(a,c)` 和采用的总输出 `c`。现有 `drafted_tokens` / `spec_decode_draft_tokens_total` 在对外报告中沿用“送验”含义。新增生成数另列。 |
| 行级分布 | 对第 `1..v` 位记 attempted，对第 `1..a` 位记 accepted。硬上限为零，或策略希望投机却因容量被裁为零，计入 `fallback_steps` 并区分原因。硬上限为正且策略主动选零，另计 `policy_bypass_steps`。部分缩短记录裁掉的列数。 |
| Executor 的实际工作 | `ExecutionWork.rows=B`，decode 的 `scheduled_positions=N`。另外记录物理 target 列、布局、backend 步/列和 owner token 负载。每个 ticket 只纳入一个完整 step 样本，取消仍计实际已发生工作，失效样本不参与策略训练。 |
| 产品与工具 | 同时更新 typed stats、JSONL、monitor/metrics、CLI 与 bench report。核对 serve runner 和 Dashboard 的分母。旧对外 timing 字段遵守已有语义。长期曲线由 Dashboard 采集端保存。 |

[公共观测类型](../../include/ninfer/observability/execution.h) · [Engine 汇总](../../src/runtime/engine/engine_execution.inl) · [Serve metrics](../../src/serve/metrics.cpp) · [Bench report](../../bench/inference/report.cpp) · [Dashboard 消费端](../../dashboard/server/ninfer.ts)

### 最终验收

| 问题 | 证据及判断方式 |
|---|---|
| 调整长度有没有引入采样偏差？ | 以两三个有效 ID 独立枚举 one-hot 和 sparse `q` 的接受、残差、bonus，将该支持域放进生产词表尺寸，包含跨分片 ID、mask 和 penalty。固定相同 represented logits、`q` 和逻辑事件，换物理行应使用相同 RNG 键。真实模型的不同浮点分派按 Op 数值合同评估。 |
| 所有权与恢复是否完整？ | 复用 S2 的资源压力和 S4 的跨轮序列。加入受限与普通请求混批、前缀分叉、Snapshot/Replay、空 owner、退役前取消和 lane 换绑。检查无重复发布、无未授权覆盖、回收后可持续推进。 |
| 自动策略能否改善使用体验？ | 同一模型产物/后端/head、KV、Graph 及采样，对比 adaptive、该负载最佳固定深度、已加载 backend 的 ordinary。固定 DFlash block 比 target 深度。另行报告改变 block 配置的实验。使用 75 条混合语料、Story 与结构化高接受场景，C1 看每请求吞吐与 step，C8 看目标 batch step、实际总吞吐、请求延迟和 corpus makespan。 |
| 是否为收益付出了过多常驻或控制开销？ | 同时报每 rank 顶层图数、启动时间、workspace、Context 余量，以及提交前每批策略/容量规划的 CPU 时间。按合理重复次数给出波动，报告没有收益的部署或负载。 |
| 切换能否收尾？ | 三套模型和两个构建 target 完成完整构建。在已声明组合上实际运行相应路径。删除已替换的固定宽度派生、临时强制逻辑、重复字段和过渡接口。复核生产调用与 wire、预算、观测、文档使用同一合同。 |

性能结论分两次作出。先在相同负载、固定满窗和相同硬件上对比迁移前后的引擎，检验既有快路及资源成本。再在新实现上比较最佳固定档与自适应，使用独立于调参的请求/seed，记录相同采样口径下的实际输出长度、累计接受量、目标 batch 的 step、单请求 P50/P95 延迟与服务 makespan。短窗微测用多轮独立的稳定窗口估计波动。服务数据沿用发布口径统计 reasoning，明确 control、prefill 和恢复工作的归属。

高接受负载应能选回有竞争力的满窗快路，混合负载应有超出测量波动的实际净收益。两档难以区分时按实测持平处理。若选择器、额外图、映射或 history 修补抵消了收益，在对应阶段修正或淘汰该物理快路。只有数值、状态、资源和实测均通过的部署才在日常配置启用 adaptive。[性能测量口径](../performance/methodology.md)
