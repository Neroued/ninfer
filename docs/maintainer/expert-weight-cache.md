# Expert 权重来源与 GPU 副本缓存

本文约定已实现的 bank 视图、放置加载、GPU 副本容器与 MoE 计算交接。
数学与量化布局由 [artifact 格式](artifact-container.md)和各 Op 合同拥有；
执行与生命周期由 [Engine 架构](engine-architecture.md)拥有。

Engine 支持常驻 GPU，以及受显存预算约束的 MoE 混合执行。CPU `routed_swiglu` Op
支持 Q4 GU + Q5/Q6 down 及 NVFP4 GU/down，直接消费 Host bank 并产生 FP32 分项。
GPU 使用同一组 logical expert ID，既可以计算固定子集，也可以消费补齐后的整层 bank。
Qwen3.5 MoE 与 Qwen4Exp Main 均使用这套公共调度，支持 eager 和分段 CUDA Graph。

## 来源与身份

`core/weight_bank.h` 的 `WeightBankLayout` 描述一个 affine 矩阵集合，包含本地 N/K、
矩阵数量，以及 codes/high/scales 各 plane 的 parent 偏移、单矩阵字节数和矩阵步长。
`WeightBankRegion` 显式指定展平 parent 上的起始行、矩阵行数、数量和行步长。
几何来自 `WeightGeometry`；bank 不另行定义 codec。

`WeightBankView` 借用 ModelShard 的不可变 `WeightParent`。`matrix_weight` 可以取得一个
原生矩阵视图，包括每次 Use 独立传入的 activation divisor。矩阵字节数与步长允许不同：
例如 GU bank 中所有 gate 构成的视图，矩阵行数是 I、步长是 2I。

- Groupwise INT 使用已有 rank-2 parent `[E*N,K]`，E/N 由模型 Binding 明确给出。
- NVFP4 使用 rank-3 `[E,N,K]`，codes、swizzled G16 scales、FP32 divisors 分别成 plane。
  切片保持 128 行边界，每个本地矩阵位于同一 divisor 域。
- Block-FP8 使用 rank-2/3 codes 和 BF16 `[E,N/128,K/128]` scales；不含 matrix divisor。
- Paged NF4 由其按行读取合同管理，不表示成 affine expert bank。

`ExpertBankSource` 将同一 logical expert 的 GU/down 两个 bank 组合，并为 Host 来源提供
对应 parent 的 `CudaHostView`。CPU 使用数学视图的 Host 地址，上传使用注册后的传输视图。
`ExpertKey={ExpertBankId,logical_expert}` 不随缓存分配变化。
Qwen3.5 的 bank ID 按 Main 层顺序建立，选中的 MTP bank 追加在末尾。
模型 loader 逐个核对 gate/up/down Binding 的 parent、起点和形状，再建立 bank；
不从物理对象 ID 或参数名称猜测编码。

## 逻辑加载与放置

加载顺序是 `read_definition → resolve_load → materialize_model`：

1. `Binder` 检查逻辑引用、Use 和所选资源；辅助值按对应元素区间取得 owning Host 值。
   `ModelDefinition` 持有这些已读 Host 资源及其 `PreparedResources`，Frontend 借用其稳定地址。
2. `ModelDefinition` 保留配置、逻辑引用、Use 和 expert source 描述，提供无 backing 的权重几何。
   Engine 显式建立不可变 `ModelProfile`，在读取大权重前查询
   实际 Op 的 workspace 并联合解析容量。纯逻辑 binder 不承担执行布局约束。
   `resolve_load` 产生一次本地 materialization 的 `ResolvedLoadPlan`，指定目标 GPU，
   以及 Main experts 使用 GPU 或 CPU 来源。
   MTP、router、shared expert 和其余固定权重仍在 GPU；放置沿显式 WeightId 展开。
3. `plan_materialization` 按 `(physical object, Device)` 去重，并用 `LayoutBuilder`
   分别规划 Host/Device backing。Host 的 `CpuOnly` 与 `CudaTransfer` 需求合并为一份来源；
   注册对象集中在 Host backing 的对齐前缀，普通 CPU 对象使用其后的独立区域。
   同一 parent 同时需要 CPU/GPU 两份 backing 时，两份容量均计入。
4. Materializer 为 CPU tensor 分配可缓存的 `HostBuffer`，为 GPU tensor 分配
   `DeviceBuffer`。仅供 CPU 的对齐 payload 由 direct I/O 直接写入最终 Host backing；
   边界页和 CPU/GPU 双端需求使用有界 staging，一次读取后分发。
   Host 放置保留文件的页内偏移；大块 Host backing 在读取前并行触页，`CudaTransfer`
   区域随后由 `CudaHostRegistration` 长期注册，仍是可缓存的 CPU 内存。
   ModelShard 持有 allocation 与 registration；RankProgram 结束使用后，先注销，再释放 Host backing。
   仅有 `CpuOnly` 需求的加载不创建 CUDA stream、event 或 pinned buffer。
5. `bind_view(reference, backing, device)` 明确选择实际位置。
   ModelShard 持有最终 backing，所有原生权重与 bank 视图借用它，不依赖 Reader 生命周期。

这里保留完整 parent 原编码，不运行期 repack，也不把 Host 来源保存成大量单 expert 对象。
`Parameters` 保存 bank 身份、矩阵 profile 与 Use 元数据，可以描述 CPU 来源。
workspace 查询消费数学 profile；完整 GPU FFN 显式接受 resident/cached CUDA bank，
混合单层执行另外借用原生 Host bank 和交接资源。
Qwen3.5 MoE 与 Qwen4Exp 均支持完整 GPU 放置；5090 预算不足时可选择 Host Main experts
和 RankProgram GPU cache。Spark 使用完整 GPU 放置。Qwen4Exp 的 block-FP8 MTP experts 常驻 GPU，
模型编排见 [Qwen4Exp 执行](qwen4_exp-execution.md)。

## 副本容量与 GPU 寻址

`describe_expert_cache` 先返回无指针的布局及容量：数据池 C、各 bank entry 布局、
Host 目录条目数、GPU offset/divisor 表和 pinned 索引快照容量。
`ExpertCache` 随 RankProgram 建立；调用者显式提供 upload execution view，并在 acquire/release
提供 consumer execution view。ModelShard 来源、DeviceContext 和被借用的 streams 必须比缓存活得更久。
全 resident 来源要求 C=0，不创建动态 offset/divisor/上传槽存储。
注册来源的 Host 容量由 ModelShard 计费；注册字节数是其中的子集，不再加一份权重容量。

CPU 来源的每个 entry 是完整 expert 的 GU/down 数据块，只包含 codes/high/scales。
各 plane 至少 16-byte 对齐，entry 按 256-byte 对齐；不同 bank 使用自己的 entry 大小。
所有 bank 共用一个 byte pool；RankProgram 显式规划固定区和可复用区域，不把小 expert 填充成最大 expert。
entry 不是序列化 parent，不追加虚假的 divisor trailer。

NVFP4 的 matrix divisor 是静态元数据：CPU 来源在启动时镜像到 GPU，常驻 GPU 来源
直接引用原 divisor plane。Host 元数据与 payload 的 divisor 必须表示同一组值，
由来源建立者保证；artifact materializer 从原 payload 提取并验证。
运行期上传只搬 codes/high/scales，divisor 始终按 logical expert ID 寻址。

`device_weight_bank` / `device_expert_bank` 从原生 CUDA bank 建立借用视图，不分配或重排权重。
`ExpertBankDeviceView` 同时记录逻辑 expert 数和 CUDA device。
`ExpertMatrixDeviceView` 的寻址合同：

```text
resident: offsets == nullptr; plane_address = plane_base + expert * plane_stride
cached:   offsets != nullptr; plane_address = plane_base + offsets[expert]
divisor_address = divisor_base + expert * divisor_stride
```

缓存的 plane_base 是 pool_base 加该 bank 的 plane 偏移；`indexed_storage` 描述整个被借用的
pool 范围，用于检查输出和 scratch 不与权重重叠。offset table、pool 和 divisor
分配地址终身稳定，可供 Graph 捕获。`UINT64_MAX` 是未发布哨兵；消费者只可读取
acquire 成功所保护的 keys，不能把表里其他条目的内容当作可用性保证。

当前真实权重的编码数据量如下，均不包含分配器及执行 workspace：

| 来源 | 单 expert 动态数据 | 完整 bank 动态数据 |
|---|---:|---:|
| 35B Q4 GU + Q5 down，256 experts | 1,802,240 B | 461,373,440 B |
| 35B Q4 GU + Q6 down，256 experts | 1,933,312 B | 494,927,872 B |
| Flash Next NVFP4 GU + down，512 experts | 2,764,800 B | 1,415,577,600 B |

Flash Next Main 48 个 bank 的两组 divisor 共 196,608 B；offset 表同为 196,608 B。
其 MTP 原始 block-FP8 expert payload 为 2,516,889,600 B，按固定 GPU 来源计入容量。

## 准备、发布和消费

`ExpertPlacement` 指定 logical key 及其在区域内的 256-byte 对齐偏移。构造时安装
`[0,fixed_bytes)` 的固定布局；固定条目在缓存生命周期内不能替换。`replace(region,entries)`
完整重建一个可复用区域：新布局不能重叠、切断已有 entry 或借用区域外的同一 key。
调用者先 release 旧消费者；上传流等待其完成事件后覆盖 payload 与索引。未提交的旧上传被取消，
已提交上传通过同一条流排序。替换不搬移区域外副本，也不执行权重 D2D。
替换的 reader wait 当即排入上传流。RankProgram 必须先提交当前层的完整上传及发布事件，
再替换后续预取窗口，避免后续窗口的等待挡住尚未提交的当前层数据。

`prepare(ExpertCacheRequest)` 选择已有安置的 required 和 prefetch，重复 key 只计一次。
它只改变上传需求，保留所选条目的地址与跨批进度。布局、窗口轮转和保留集合由 RankProgram 决定。

`progress(ExpertUploadBudget)` 按本次 payload 字节数与批次数预算提交工作，不显式等待 GPU。
启动配置同时限制单批 payload、复制描述符数和在途批次数；零预算只检查完成状态。
返回值包含 required 是否已发布、prefetch 是否已经就绪、是否等待上传槽，以及本次提交的批数和字节数。

权重直接从注册来源复制，不经过 CPU packing。较大的连续 bank 区间使用 pitched H2D，
小集合与离散区间使用批量 H2D；选择仅依赖实际地址和规模。传输描述符预分配并复用。
普通 Host 地址与 CUDA 映射地址分别保存，批量复制消费后者，CPU 数学继续消费前者。
一个 expert 可以跨批次；完整 payload 提交后才发布它的 offset。编码内 padding 保留原字节，
entry 对齐空隙不计入传输 payload。

上传槽仅持有描述符、dirty index 快照与完成事件。数据批次之后提交索引批次；
完成前不重写该槽的快照和描述符。Host 来源没有有效注册视图时，在建立缓存前拒绝。

发布顺序：

```text
encoded data DMA → changed offset entries DMA → slot completion event
    → acquire: consumer stream wait → kernels / Graph replay
    → release: consumer completion event → region replacement
```

索引只更新已可安全修改的条目，可合并相邻 dirty 条目，不覆盖正在被其他 use 读取的表项。
每个已发布 entry 记录上传槽及发布代次；acquire 只等待所需代次，已回收的旧槽代次意味着
该次上传已完成。后续无关预取不会成为旧数据消费者的新依赖。

`acquire(keys,execution)` 在发布已经排入流后即可成功，不要求 Host 已观察到 DMA 完成。
未发布时返回空；已 release 的 use 占满有界环时返回空，由调用方等待回收；全部槽仍有未 release 的 use 属于调用错误。
成功取得的每个 use 都须在同一 consumer stream 调用一次 `release`，包括 resident bank。
release 必须位于最后一次真实 kernel/Graph replay 之后；这两个交接操作在 capture 外执行。
区域替换在上传流等待已 release 的 consumer event，再更新 payload 和 offset；
尚未 release 的 use 阻止替换。use 记录副本代次，旧 use 完成不会减少新副本的 pin。
Host 的 release 不代表 GPU 已完成读取。CUDA 提交失败后停止接受工作，销毁前排空
已提交传输和消费者。

缓存不包含 CPU 线程池、不执行 MoE 数学，也不拥有模型执行策略。
上层通过 required/prefetch 和 progress 预算表达实际需求；已经提交的复制不能撤回。
注册及首次上传的准备成本在启动阶段处理。CPU 计算与 DMA 可以同时读取不可变来源，
但共享系统内存带宽；RankProgram 为权重上传和小 bridge 分配独立 streams。

## MoE 计算交接

数学、输入输出与 alias 合同由 [MoE Op](../../include/ninfer/ops/moe.h) 和
[SumContributions](../../include/ninfer/ops/sum_contributions.h) 拥有。

- `topk_softmax_route` 写出 logical expert IDs、原始 top-k softmax 系数和独立 shared sigmoid
  系数。路由张量为 I32/FP32 `[K,T]`，shared 系数为 FP32 `[T]`。
- `routed_swiglu` 消费同一次路由，可以计算 All、None、mask 的 1 侧或 0 侧；不重新归一化。
  mask 表示计算分工，GPU offset table 表示物理位置。两者独立，未选 expert 的偏移和权重不被读取。
  可选 shared term 在 GPU 上计算一次，也适用于 None。
- CPU 分支写 FP32 `[H,T]`；GPU 分项可写 FP32，最终计算可写 BF16，或在一次最终舍入前加上原 BF16 residual。
  `sum_contributions` 接受一到两份 FP32 分项与可选 BF16 residual，产生 BF16 输出。
- `topk_softmax_swiglu_moe` 定义完整 router、routed/shared SwiGLU 与可选 residual 的融合计算。
  它与独立 Op 共用 kernel 机制，保有自己的格式/shape 分派和调优策略；全 GPU Main/MTP
  通过这一入口执行。

CUDA 数学支持 35B 的整数 expert profiles，以及 Flash Next 的 NVFP4、block-FP8 profiles，
通过同一组 [MoE 语义入口](../../include/ninfer/ops/moe.h)消费实际表示与 activation Use。
CPU 支持 Q4+Q5、Q4+Q6 和 Main NVFP4，输入为 BF16，
SiLU 与 down 输入保持 FP32，不量化激活；CPU block-FP8 尚未实现。
`CudaSwiGLUExperts` 借用 affine 或 indexed GPU bank；`CpuSwiGLUExperts` 借用完整原生 Host bank，
直接按各 plane 的偏移和步长消费，NVFP4 使用每个 expert 自己的 matrix divisor。
CPU 权重接口不携带 activation divisor。GPU activation divisor 属于 Use，matrix divisor 属于权重表示，二者不互相替代。
融合 GU/down 分别取相关 Use 的 activation policy 交集；NVFP4 A4 需要共同的正值 activation divisor。

Op 不分配长期资源或上传权重；CUDA 分支消费 Device 路由，CPU 分支同步消费 Host 路由。
RankProgram 提供 scratch 和执行 view，负责发布、acquire/release 及 bridge 完成依赖。
固定 profile 的 CUDA Graph 可以重放新的路由、mask 和
索引内容；更新必须发生在旧消费者结束之后。shared 始终驻留 GPU，不进入动态 expert 集合。

`models/common/moe` 的单层执行借用 `moe::ExecutionView`。`moe::Partition` 从本层已就绪 GPU ID
快照建立互补 mask 与 cache keys，CPU/GPU 在本次调用消费同一快照。mask 在计算流更新，
pinned staging 在 DMA 完成前不复用。`moe::Bridge` 持有 CPU team、普通 Host scratch、
pinned/Device frame，以及独立的 bridge stream/events。资源查询分别返回两端 frame 和
scratch 上界，可对多个 profile 取最大值；GPU scratch 由调用者提供，route/routed 顺序复用。
纯 GPU 分工不建立 CPU team 或 bridge，完整融合调用显式接受 resident/indexed bank。
公共执行支持 `Write` 和 `AddResidual`；前者不读取旧输出，后者在最终舍入前加入已有
BF16 输出。Qwen3.5 使用后者，HC 架构可消费前者的独立 block 输出。

调用方先用 cache `prepare/progress` 发布 required。混合调用先在计算流提交 route，
回传 hidden 和连续的 IDs/coefficients，再提交 GPU routed/shared。小 T 的复制直接按计算流
排序，较大 T 使用独立 bridge 流及事件，使传输与 GPU 计算重叠。调用方持有 GPU 分工的 lease；
RankWorker 等输入就绪后运行同步 CPU collective，与 GPU 分支重叠。
CPU FP32 分项通过同一传输流上传，计算流按依赖合并两份分项与 residual。正常路径只等待 CPU 输入，
最终输出按计算流排序。一份 frame 由同一 RankWorker 在固定 streams 上连续复用；下一次
route/D2H 经过上一层 combine，因而不会覆盖仍在上传的 Host 分项。资源销毁排空两条流。
若本次路由没有 CPU assignment，跳过 CPU collective 和分项回传。单层 cache 交接在 capture 外执行。

RankProgram 记录每次完整 traversal 中实际参与计算的 CPU 层数；连续相同 T 下，CPU 频繁参与时
在层间保持团队轮询，稀疏参与时允许休眠。启停使用不同阈值，T 改变或 traversal 未完成时
重新观察。该策略不依赖 expert 的具体 ID、GPU 命中率假设或请求阶段，作用域在 traversal
结束时释放；完整 GPU 流式计算不保持 CPU 团队活跃。

## Engine 预算与逐层执行

`NINFER_VRAM_BUDGET_MIB` 是引擎可使用的 Device 容量上限，另受启动时实际空闲显存限制。
预算包含固定权重、RankProgram State/KV、workspace、cache、offset/divisor 表、bridge/masks
及运行时余量。自动 KV 先保证一个完整上下文；Host 模式把其余容量优先用于 expert。
显式 KV 配置仍完整兑现。Host 来源、注册范围和 pinned frame 单独报告，注册范围是
Host backing 的子集。MTP、router、shared、Vision 和 draft 始终留在 GPU。

两种架构的完整 GPU 放置可行时使用融合路径，不创建 Host expert 副本；Host 路径保留完整
原编码来源。
`EngineOptions::moe.expert_priority` 可指定每个 Main 层全部 logical ID 的优先级排列。
35B-A3B 和 Flash Next 配置默认使用模型私有的静态标定排序，其他配置使用 ID 顺序；
显式排列覆盖默认值。排序只决定缓存放置，不改变 router、coefficients 或 expert 数学，
也不在请求期间更新。标定来源说明与数组分别保存在各模型的 `moe_priority.cpp` 和
`expert_priority.cpp`；同配置的其他训练权重可提供自己的排序。
各层按字节均衡分配额度，以完整 expert 为单位选取优先级前缀；排列本身
不强制启用 offload。优先级影响目标集合，实际 GPU 分工由已就绪副本决定。
`moe.cpu_workers=0` 自动选择物理核并为执行调用线程留核；非零指定团队规模。

`moe::Schedule` 由 RankProgram 持有，封装 offload traversal、CPU 活动和图交接；其具体
`offload::Resources`、CPU/team 计划与分段图只在提供 offload 的目标编译。
Resident 调用直接消费 ModelShard bank，复用完整 GPU 融合 Op。公共 plan 按实际 MoE 调用顺序组织，
bank ID 与模型层号独立。模型提供 Main/MTP 的 bank 放置与优先级；混合计算阈值及参考 CPU 团队规模作为
计算 profile 的策略输入，缓存规划不固定任何模型几何。扣除固定资源、KV、workspace、bridge/masks 和索引后，
剩余容量 C 分成固定区 R 和流式窗口。启动时联合求解均衡的固定额度及窗口容量：每层完整
bank 为 W_l，固定 expert 为 R_l，每个窗口必须容纳 `max_l(W_l-R_l)`。不先扣除两个完整 bank。
规划枚举完整 expert 的额度边界，并利用取整余量；固定集合取各层优先级前缀，数量和地址终身稳定。
默认使用两个窗口；预算不足双窗口时使用单窗口，连单窗口也不可行时拒绝该容量计划。

小 T 把全部窗口空间合并为一段连续额外缓存，继续按均衡字节额度扩展每层的优先级前缀。
固定布局、额外布局及各层缺失集合都在启动时确定；初始固定与额外上传完成后才进入执行。
每次 Main traversal 只依据实际 T 形成需求：

- 小 T：GPU/CPU 消费同一 ready 集合及其补集。稳定时复用快照和 lease，无权重上传。
  从大 T 返回时，在层间按层序分批恢复额外布局；无需等待全部恢复，未 ready 的 expert 由 CPU
  计算。每个完整 expert 上传完成后才进入快照，恢复完成后重新稳定。
- 大 T：窗口紧凑装载当前层的缺失集合，与固定副本合成完整 indexed bank；另一窗口预取下一层。
  当前数据先单独发布完成事件，再推进下一层上传。窗口最后读取之后可排入下一次覆盖，
  单窗口模式按相同依赖顺序加载。不同层缺失字节数导致的窗口尾部余量暂不使用。

prefill、Replay、ordinary/control 和 speculative verification 共用该入口，不依据请求阶段
改变 MoE 策略。权重上传与当前层计算及下一层 mixer 重叠；连续大 T 不在 chunk 边界恢复
小 T 布局。大小 T 切换只重建窗口，不改变固定区；一次完整额外缓存恢复的 H2D 不超过窗口总容量。
恢复途中再次进入大 T 时，取消未提交恢复，并在已有上传之后重建流式布局。
cache 不进入 StateImage、checkpoint 或 speculative 回滚。

MemorySummary 报告池容量、固定区、窗口总容量与数量、实际占用、完整小 T 目标和实际 ready
字节及逐层数量。固定区加窗口等于 cache_bytes；ready 与 target 是其中的状态或目标，不能重复相加。
另报告 H2D 总量及其中的预取字节，不再提供碎片整理计数。

`prefill_chunk=0` 表示未指定：常驻模型默认 1024，CPU expert offload 默认 4096。
`--prefill-chunk N` 接受 128 的正整数倍，启动时按指定值或固定默认值规划容量，实际调用
受 context 剩余长度限制。测量签名包含优先级、各层缓存额度、团队、池容量和 chunk，
不把常驻模型的重算成本标定直接套给 offload。

Qwen3.5 常驻执行使用整段 CUDA Graph；Main offload 使用分段 Graph，也保留 eager 提交。
稳定小 T 将 mixer 与 MoE GPU 前段合并；窗口流式加载或额外缓存恢复期间分开提交，
在 mixer 之后取得快照和加入权重就绪依赖。CPU 等待图中的输入就绪 external event，
与 GPU routed/shared 并行计算；本轮没有 CPU contribution 时跳过 collective 和回传。
捕获路径把输入 D2H 排在 GPU experts 前，避免独立拷贝分支被延后；较大 T 的结果 H2D
仍使用 bridge stream。Eager 保留较大 T 的输入拷贝分支。
缓存交接和预取留在图外，lease 释放排在最后一个 GPU 权重消费者之后。

MTP 的 Main verification 和 DFlash 的 target 使用同一分段调度；MTP 自身及 draft
延续常驻 Graph。相同 exact-B 的 target executable 在 draft 的不同拓扑间共享，
每次选择 profile 同时安装对应 target envelope；MoE 段不随 context 档位复制。
约束 mask 在首个 target GPU 段提交后准备。Graph 实例与 bridge/masks
均由 RankProgram 持有。启动校准借用已加载的 Host bank，以单层窗口承载权重，但按候选正式计划
捕获 All/None/MaskOnes 和 steady 段；CPU/GPU 计算仍使用同一组执行原语。
expert ID 与 offset 改变不要求重新校准，计算变体集合改变时校准对应集合。
销毁临时资源后扣除重建估计，再确定 expert cache 额度；销毁时先结束设备
工作并释放 Graph，再释放其引用的事件和存储。Prefill、Replay 延续 eager 提交，MoE
计算策略仍只依据实际 T。

测试与实测入口分别见 [Tests](../../tests/README.md) 和 [Benchmarks](../../bench/README.md)。
