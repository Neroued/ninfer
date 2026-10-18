# Qwen4Exp 执行与状态

本文是 Qwen3.8-Flash-Next 在 NInfer 中的执行支持入口，维护层连接、资源所有权、
PLE 流水、MTP/Vision、CUDA Graph 和完整 continuation 的提交规则。
模型数学、量化和持久布局见[模型与权重](qwen4_exp-model.md)。
请求调度与发布由 [Engine 架构](engine-architecture.md)维护，通用
[上下文缓存](resource-scheduling-and-context-cache.md)、
[Paged KV](paged-kv-cache.md)及 [expert cache](expert-weight-cache.md)各自维护其公共合同。

## 1. 当前支持范围

| 能力 | 当前实现 |
|---|---|
| Text | 公共 Engine 的 prefill、ordinary decode、forced control、Replay 和取消 |
| 并发与上下文 | 启动固定 1–8 路 resident lanes；本实例最大上下文 262144，实际容量由启动预算决定 |
| Vision | 图像、视频、多媒体消息；共享 Vision 编码器、MRoPE 和跨 Text chunk 特征交接 |
| 原生 MTP | draft 数 1–5，支持 Text/Vision、1–8 路、采样 penalty 和约束输出 |
| CUDA Graph | resident Main 整图、MTP draft chain、独立采样/接受图、无 head 的单列 control 图；expert offload 保留分段交接 |
| KV 存储 | BF16、INT8-G64、FP8-row256、NVFP4-G16、K8V4；index keys 固定 BF16 |
| 上下文复用 | 完整 StateImage 与 Main/MTP 页组、Device 前缀命中、Snapshot/Replay；5090 还提供 Host 副本 |
| 前端与服务 | thinking、thinking budget、tool call、GBNF/JSON/schema/choice/regex；共用 Qwen Frontend 和现有 HTTP 协议 |
| CausalScoring | 公共 Engine 的单路 Text 评分；不启用 Vision 或 speculative backend |
| Expert 放置 | Main 支持 GPU resident；5090 还提供 registered Host backing、GPU 缓存及整层流式执行；MTP/shared 固定 GPU |
| PLE | V3 磁盘表、Host hash/NF4 解码、有界行缓存、增量准备和双槽 H2D |

当前限制：

- Speculative backend 仅支持原生 MTP，使用完整共享 BF16 head；不支持 DFlash、DFlash2
  或 indexed proposal head，因此不使用 `--lm-head-draft`。
- Offload 模式保留完整 Host expert 来源。GPU 缓存没有请求自适应排序或 decode 动态换入；
  小批量缺失 expert 由 CPU 计算，没有按需换入 GPU 计算的第三条分工路径。
- PLE 当前只有磁盘准备路径，没有整表 RAM/GPU lookup。
- Prefill、Replay、Vision encode、MTP 历史对齐与状态采用均使用 eager。
  Resident Main 图包含全部层和 PLE 消费；PLE Host 准备在提交图前交付输入，offload 保留 CPU expert 交接。

用户命令见 [CLI](../cli.md)与[Serving](../serving.md)，产物生成见
[转换指南](../weight-conversion.md)。这些能力描述当前实现；公开的吞吐与质量结果由
[性能文档](../performance.md)和 model cards 单独维护。

## 2. 轴、前沿与所有权

| 符号 | 含义 |
|---|---|
| T / B | 每行物理输入列数 / 本次紧凑 batch 行数 |
| M | `T*B`，展平列号 `t+T*b`；Linear、HC 和 MoE 使用此轴 |
| n[b] | 本窗口有效输入前缀长度，Device `valid_columns[B]` |
| F / F0 | 已执行输入前沿 / 候选窗口基准前沿，均按原 token 计数 |
| a[b] | 最终采用的输入前缀长度，不等于预取、采样或候选总长度 |
| State slot | StateImage 的物理槽；独立于 resident lane、batch row 和 KV table row |

规范边界下 Main 前沿为 F，启用 MTP 时 Backend 前沿为 `max(F-1,0)`。
Backend 位置 j 配对 Main conditioning j 与输入 embedding j+1，RoPE 使用位置 j 的坐标。
Ledger 可比 Main 多一枚已经发布、尚未执行的 anchor；I/O 预取不推进任何模型前沿。

HC 的四条 streams 是特征轴，GDN/QSA/MoE 的 token 数不乘四。
只有 MTP 的共享 fc_hidden 把四条 stream 作为独立 Linear 列，使用 4M 列。
普通 `[H,T,B]` 与 `[H,M]` 可直接改 view；parent 子范围必须保留实际 strides。

| 所有者 | 内容 |
|---|---|
| ModelDefinition | Config、Binding/Use、表示几何、Frontend 资源和 PLE hash controls |
| ModelShard / Parameters | 本地不可变 Host/GPU 权重、PLE 文件 backing 和实际 Op 参数视图 |
| Session / ContextDirectory | 请求 continuation、checkpoint、Main/MTP frontiers、RoPE delta、PLE raw-ID tail、state/page/table/Host 范围分配与引用 |
| RankProgram | StateImage 和 Main/MTP KV 的固定 backing、records、workspace、PLE 服务与上传槽、Graphs；offload 另持有 expert cache/CPU bridge |
| Ops | 借用的输入输出和 scratch；计算或状态提取，不发布请求状态 |
| Runtime / Frontend | 请求调度、保留策略、输出 preview/commit 和协议语义 |

HC/Linear 可计算 padding 物理列；RankProgram 提供有限安全输入。
QSA/PLE 允许零有效列，GDN 的 active batch 不包含空行。Main 在完整 forward 前确定
紧凑成员，保持各层、records 和 commit selectors 的行对应关系。

## 3. Main 层调用

```text
embedding + 可选视觉替换
  → hc_expand_norm：residual、normalized
  → attention hc_mix：block input、FP32 injection
  → GDN 或 QSA：block output
  → hc_combine_norm：新 residual、MoE normalized
  → MoE hc_mix → routed + shared MoE
  → hc_combine_norm：新 residual、下一层 attention normalized
    （末层改用 final norm）
  → final hc_mix → sample hidden → head
```

Combine 与下一次 norm 融合，但 norm 消费实际 BF16 residual。
进入 PLE 前先通过 `hc_combine` 物化完整 residual；PLE 输出 residual 和本层
attention normalized 后，继续相同层流程。Main 最后一层保留的 conditioning
是 final norm/mixer 之前的 residual；不是 normalized 或 block output。

### 3.1 GDN

AB 投影经 `gdn_control_proj` 产生 FP32 g/beta，QKVZ 经下表的输入路径产生 Q/K/V/Z。
Recurrence 使用 `normalize_qk=true`、`scale=1/sqrt(128)`，
读取当前 token 更新后的 FP32 state；随后做 sigmoid-gated RMSNorm 和 output Linear。

| 路径 | 输入组合 | 状态处理 |
|---|---|---|
| Prefill / Replay / scoring | B=1；`linear_split → causal_conv1d_silu_split → gated_delta_net` | 保存整个窗口末端 conv/recurrent state |
| Ordinary / 单列 control | T=1，B=1..8；`gdn_input_proj_conv_snapshot → gated_delta_net_batch_update` | 写候选 state slot |
| MTP target verify | T=K+1；`gdn_input_proj_conv_record → gated_delta_net_replay_record` | 基准只读，保存实际 records，采用时 exact fold |

Z 绕过卷积，保留到输出 gate。独立 LinearSplit 的 BF16 出口是公共边界；
融合 projection/conv 不增加私有 BF16 投影中间边界。

Verify records 包含卷积前最新 history 列、卷积/SiLU 后且 L2 norm 前的 K/V、
以及实际消费的 FP32 g/beta。`GdnReplayFoldPlan` 从同一次基准 state 重放接受前缀；
结果逐 bit 对齐同一物理 record-producing block 的 snapshot。
不得缩短 T 重新跑投影来替代 fold。原物理 B、row 身份、record 布局在采用前保持不变。
详见 [GDN input](../../include/ninfer/ops/gdn_input_proj.h)与
[ReplaySSM](replayssm-gdn.md)。

### 3.2 QSA

Main 的 Q8 Q/gate parent 与 BF16 KV/index parent 分别调用 LinearSplit；
MTP 使用一个 BF16 joint parent。投影直接产出连续分段，随后依次调用：

| 阶段 | 交接 |
|---|---|
| `qsa_prepare` | projection、norm/RoPE、旧 raw tail → BF16 Q/index-Q、主 KV、完整块 index keys、raw-key/坐标 records |
| `qsa_indexer` | index-Q 和 compressed cache → Device I32 block IDs `[512,T,B]`、counts `[T,B]` |
| `qsa_sparse_gqa` | Q、原 gate logits、selection、主 KV → gated BF16 output，再交 O Linear |

Selection 是原 token 域的四-token 逻辑块号，与物理 page ID 无关。
Sparse GQA 直接消费前 count 个 IDs，并补上当前不完整块的可见尾部，不建立展开 token
列表或全局 gathered KV。每 query 只读取自身因果范围，候选后缀即使已经写入也不可见。
可见完整块数 C≤512 时，Indexer 在同一 kernel 内全选并跳过评分。

压缩使用 raw index-K；跨窗口先拼旧 tail，再做四列平均，保留规定的 BF16 均值舍入，
之后 norm/RoPE。块首旋转坐标来自实际 MRoPE，不能由 token ordinal 猜测。
主 K 的 norm/RoPE 结果直接进入存储 codec，不额外舍入为 BF16。
Sparse GQA 保持 A16 计算，不将 Q 或 softmax probabilities 量化为低比特；
量化 KV 在片上解码，必要的值域逆变换和完整归约后才乘 gate。
精确公式、codec 与读写域见 [QSA](../../include/ninfer/ops/qsa.h)。

### 3.3 MoE

MoE 消费 HC 输出的 `[2560,M]`，使用 `MoeEpilogue::Write` 产生独立 block 输出。
Router/shared parent 的持久合并不强制一次计算；shared 始终在 GPU 上计入一次。
GPU 的完整融合与 routed subset 入口均已实现，CPU subset 产生 FP32 contribution，
由公共 mixed execution 汇合并最终舍入。

Resident 路径直接调用完整 GPU 融合 MoE，bank 地址由 ModelShard backing 保持稳定。
Offload 路径由公共 `models/common/moe` 按实际 M 选择策略：

- 小 M：GPU 计算 ready experts，CPU 计算补集；稳定时不搬运权重。
- 大 M：固定缓存与窗口中补齐的缺失集合组成完整 bank，全 GPU 计算，并预取下一层。
- 切换：固定区地址和数量不变；窗口在大 M 流式区域与小 M 额外缓存之间重用。
  返回小 M 时逐层恢复，尚未 ready 的部分由 CPU 计算。

各层额度按字节均衡分配，集合为启动优先级的前缀。Flash Next 默认使用已标定的静态
逐层排序，`MoeOptions::expert_priority` 可覆盖；排序不影响 router 或 coefficients。
阈值由计算策略和启动 CPU 团队规模解析，prefill/decode/verify 名称不参与分派。
容量、上传、lease、CPU/GPU 交接和分段图的公共合同见 [Expert cache](expert-weight-cache.md)。

## 4. PLE 准备、传输与消费

### 4.1 磁盘表和读取服务

`ple::disk::Table` 接收已绑定逻辑参数，持有长期只读文件 backing、冻结区间映射与
每 shard 的 NF4 LUT。启动不扫描整表。行地址先映射到 V3 实际文件区间，再扩展到
direct-I/O 对齐范围；跨续卷或非对齐边界可以产生多段读。短读只有覆盖全部必需字节
才可接受，不能用补零替代丢失数据。

RankProgram 的 `PrepareService` 使用一个 I/O worker，执行
`hash → 行去重/cache → 物理读合并 → O_DIRECT/io_uring → 收集 → LUT 解码`。
当前配置为 64 MiB 四路原始行缓存、128 个在途读、两个窗口槽。
行缓存和在途读可跨窗口复用；必需窗口优先，预取保留必需窗口的容量。
Worker 不调用 CUDA、不修改 ledger、frontier 或 StateImage。

输出为连续 BF16 `E[2560,T,B]`；无效尾列精确置 +0。
Lookup 的 hash、NF4 精确舍入与布局见[模型文档](qwen4_exp-model.md#5-ple-表hash-与-lookup-出口)。

### 4.2 窗口、ticket 与上传槽

Ticket 由物理 slot 和 generation 构成。每行携带 sequence、epoch、起始 frontier
和 raw-ID tail；prepare/append 返回前复制 caller IDs，后续不借用可变 caller spans。

| 操作 | 作用 |
|---|---|
| prepare | 固定 T/B、行身份和已知 ID 前缀；无空槽返回暂不可准备 |
| append | 按行连续追加候选 IDs，可立即发起读取 |
| seal | 冻结最终有效输入长度；截去尾列置零，之后不可追加 |
| HostReady | 已 seal，全部有效 E 完成；尚不代表 Device 可消费 |
| upload / acquire | 在独立 transfer stream 排入 H2D 和 event，再借出本代次的 Device E、ready/consumed events |
| consumed / release | 最后 Device 消费、H2D 和 Host writers 都退役后，才重用 slot |

`UploadSlots` 独立拥有 pinned/Device E；PrepareService 借用其 Host 槽。
Consumer 在首个 E 读者前等待 ready，在最后读者后记录 consumed，再通知 owner 已提交。
Graph 将这两个依赖捕获为 external event nodes；本代次 H2D/event 必须先于 graph replay
排入 producer stream。尚未提交的下一代 event 不能作为本轮就绪信号。
RankProgram 先创建上传槽，后创建准备服务；销毁时先结束 Host writers，再排空上传和消费。
取消撤销该窗口的发布意图，仍有其他窗口使用的共享读继续完成。
晚到完成可填充正确的不可变行缓存，但不能写入重用后的 ticket。
读取失败通过当前 unit 的错误路径传播，不发布部分 E。

`PlePipeline` 组合准备、增量 ID D2H、H2D、消费和回收。
它在等待 compute 完成时继续推进已就绪窗口的上传。
Canonical state 和 accepted IDs 不借用将要释放的 ticket。

### 4.3 与 GPU 重叠

| 场景 | 当前准备时机 |
|---|---|
| 首个输入窗口 | IDs 已知后提交，可与 Vision encode、embedding 和 PLE 前层重叠 |
| Prefill / Replay | 消费当前 PLE 后准备下一个实际 Text 窗口，与后续 Main 层计算重叠 |
| Ordinary decode | anchor 已知即准备；resident Main 提交前等待 HostReady/H2D 已排入，图内在 PLE 入口等待上传完成 |
| MTP verify | 先准备 anchor；各步 draft ID 异步 D2H 后按顺序 append，全部有效 IDs 到齐后 seal |
| Control / scoring | 使用同一真实输入前沿和准备流水 |

GPU 交接为：

```text
raw IDs + raw-ID tail → PrepareService → HostReady → H2D/event → E → KV Linear → K/V
Main embedding / PLE 前层 → 完整 residual
K/V + residual → ple_gate：BF16 U、BF16 N records
residual + U + N + 旧 history → ple_conv_add_norm：BF16 residual、下一组 normalized
  → 继续 Main 层
```

PLE 卷积消费 norm(actual BF16 U) 的 N records，四个 tap 对应 -9、-6、-3、0。
Conv/SiLU 与 residual/U 相加后写 BF16，再做下一组 HC norm。
E 的最后读者是 KV Linear；RankProgram 在整个 PLE GPU 段后记录 consumed。
Resident 图中的 external record 允许 Host 在后续层执行期间回收该槽。
N records 和采用所需 IDs 保留到提交，不随 E 槽释放。
GPU 数学见 [PLE 合同](../../include/ninfer/ops/ple.h)。

## 5. MTP 的生成与采用

MTP draft 数 K 在启动时固定为 1–5；物理 target 窗口 T=K+1。
各行按剩余输出/上下文预算缩短有效长度；混合 batch 保持这一物理窗口，
整批都无法再产生 draft 时走 ordinary。
Draft 使用完整共享 head 的 argmax；target 验收支持请求的 greedy 或 sampling 配置，
并处理 penalty 与约束 mask。验收数学由
[SpeculativeAccept](../../include/ninfer/ops/speculative_accept.h)维护。

1. **历史建立。** Prefill、Replay、control 将输入 embedding 与前一列 Main conditioning
   配对。根部跳过第一列，连续 chunk 带上一列旧 conditioning。只执行 MTP stem、
   attention HC、QSA projection/prepare 和 tail 提取，不执行 attention readout、MoE 或 head。
2. **Draft。** 第一步在 Backend F-1 配对 Main conditioning 与 anchor，单独保留其
   bridge tail；后续步骤使用前一步 MTP conditioning。工作 tail/hidden 不成为 canonical state。
3. **Target。** Main 执行 `[anchor,drafts...]`，记录 GDN transitions、QSA raw keys/坐标、
   PLE N，以及 sample/conditioning 窗口。所有记录保留原 T/B 和成员顺序。
4. **采用。** 后端接受 c 个 draft，许可 c+1 个输出；Frontend 可进一步截断为 a。
   Main 采用前 a 个输入，末端两种 hidden 同取 a-1 列。Backend 从真实 bridge tail 出发，
   用 Main conditioning 与已接受 shifted IDs 重建其后 a-1 列历史，提交到 F+a-1。
5. **结算。** 正式 penalty counts 只增加最终发布的 tokens；拒绝后缀不进入有效
   State/KV coverage，余下的 correction/bonus 成为下一轮 anchor。

即使零 draft 被接受，正常一轮仍采用 anchor，即 a=1。零接受的取消/失败行不采用候选，
按 Engine 终态流程释放。Checkpoint 保存 Main 末端 pair 和已提交 Backend 状态；
恢复不依赖未决 draft 内部特征，也不重复执行 head 来重建已保存 hidden。

## 6. Vision 与前端

共享 Qwen Frontend 提供原 IDs、媒体 patches、视觉 spans 和三轴 MRoPE。
RankProgram 在 embedding 后、HC 展开前替换视觉列；MTP 历史建立也使用对应视觉 embedding。
QSA 因果顺序仍使用 token ordinal，RoPE 独立消费坐标，生成文本使用保存的 rope_delta。

一个 Text 窗口至多消费一个视觉条目，条目可以跨多个 chunk。
RankProgram 持有一份 workspace 和独立于通用计算区的视觉 handoff；各 residency 的视觉执行
按 owner/generation 借用该 handoff。Session 只保留 owning Host 输入和消费计划。编码 scratch 可以复用通用区域；
Text/decode 不覆盖有效 handoff，另一个 Vision 编码可覆盖它，后续需要时由保留 patches 重编码。
Main 与 MTP 完成对应消费后退役条目绑定，暂停/取消也退役。

已复用前缀覆盖的视觉条目不再编码。Checkpoint 不截断同一视觉条目的消费范围，
条目中途暂停通过 Replay 恢复。Vision 不新增 StateImage 字段或 KV pool；
其历史效果已经进入 Main/MTP 状态。

Converter 默认打包
[Flash Next 模板](../../tools/chat_templates/qwen3_8_flash_next.jinja)，
运行时也可用 `--chat-template` 覆盖。Thinking、reasoning effort、tool calls、结构化输出
沿公共 Frontend 实现；模型代码不处理 HTTP 协议或执行工具。
模板默认行为和覆盖方法见[转换指南](../weight-conversion.md)，
输出约束见[约束解码](constrained-decoding.md)。
采样默认值来自 artifact 的 generation_config，thinking/non-thinking 共用该组值，
用户可显式覆盖。

## 7. 完整状态、checkpoint 与恢复

### 7.1 StateImage 与 metadata

一个规范 Main 前沿 F 的状态包含：

| 数据 | 存储与含义 |
|---|---|
| GDN recurrent | 36 层 FP32 `[128,128,48]` |
| GDN conv history | 每层 BF16 `[10240,3]`，卷积前历史 |
| Main QSA tail | 12 层，各最多三列 raw BF16 index-K 和未完成块首三轴坐标 |
| PLE conv history | BF16 `[10240,9]`，最近九列 N |
| Main sample hidden | BF16 `[2560]`，来自 F-1 的 final mixer 输出 |
| Main MTP conditioning | 启用 MTP 时 BF16 `[10240]`，来自同一 F-1 的 pre-final residual |
| Backend QSA tail | 启用 MTP 时的一层 raw tail/坐标 |
| Host metadata | Main/MTP frontiers、rope_delta、最近两枚已执行 raw IDs |

模型用 `StateImageBuilder` 声明字段；Core 将每个完整 image 连续存放，Host/Device
使用相同内部布局。Raw-ID tail 保存在 continuation metadata 中，不重复放进 Device image。
F=0 的 Device state 清零、raw-ID tail 补 EOS，末端 hidden 尚无有效 token。

每个 QSA pool 的 tail 由自己的前沿 f 决定，保存 f%4 个 raw keys，其余为 +0；
完整块边界时 tail/坐标均为零。它独立于主 KV 的压缩编码，
不可用已 norm/RoPE 的 keys 代替。

### 7.2 Main/MTP 共同逻辑页组

每个逻辑页覆盖 64 个原 token：该 pool 所有层的主 K/V data/scale planes 各有 64 行，
每层 BF16 index plane 有 16 行。它们共用 page ID、block table、coverage、COW 和迁移事务。
原 token p 查 `table[p/64]`；index block g 查 `table[g/16]`。
Main 12 层属于一个 pool，MTP 的一层属于独立 Backend pool。

某页有 c 个有效原 token 时，index 只有 floor(c/4) 行有效，不能独立推进 index cursor。
Prepare 可写候选后缀，coverage 只在采用时改变。COW 保护所有可能覆盖的 planes，
Device/Host 迁移保留其原 words；恢复还必须具有同前沿的 StateImage 和 metadata。
具体行密度、Host packing 和传输规则归 [Paged KV](paged-kv-cache.md)。

### 7.3 提交与恢复边界

```text
Session 取得执行许可，预留 state / pages，形成 COW 与输入描述
  → RankProgram 执行输入准备、候选 forward 和后端验收
  → ResultReady，Frontend preview 得到接受前缀 a
  → Session 提交 Commit，取得 CommitReceipt 后发布输出
  → RankProgram 执行 GDN exact fold、QSA/PLE extract、hidden 选择、MTP 历史对齐
  → DeviceRetired，Session 结算 StateImage、各 pool coverage、输入身份和资源引用
```

`qsa_tail_extract` 从实际 raw-key/坐标 records 取得 tail；
`ple_state_extract` 从 N records 取得九列历史；raw-ID tail 按同一个 a 更新。
QSA/PLE 在 a=0 时可以向独立 destination 复制旧状态；GDN fold 的 a=0 是严格 no-op。
是否采用 destination 由 Session 的 Commit 决定，RankProgram 执行对应物理写入。

Sample 和 conditioning 必须来自同一次执行的同一接受列，按原 words 复制；
不能对更短窗口重算 final mixer。接受前沿可不对齐四-token 块或 64-token 页。
例如 F0=6、a=1 时，新的 QSA tail 是 raw 4/5/6，候选阶段写出的块 4..7 仍不可见。

长窗口只产生末端 GDN recurrent state，因此完整模型 checkpoint 对齐实际执行边界。
Session 在选定 capture 点拆分窗口；QSA/PLE 能提取中间前缀，并不单独赋予该点完整
模型恢复能力。保存 checkpoint 保留精确 HistoryPrefix，不因非页对齐而立即复制尾页；
真正写冲突时才执行 COW。

Snapshot 复制/恢复完整 image、各 pool 前缀和 metadata；Replay 重新执行已接受输入，
保留原请求身份、输出 ledger 和预算，不重复发布。暂停可以释放 lane，恢复使用新的
state slot/table row。前缀命中采用公共资源策略，不另建 QSA/PLE 专用匹配或保留算法。
不可变 PLE row cache、expert cache、ticket、scratch 和 draft 工作特征不属于 checkpoint。

## 8. CUDA Graph 与启动资源

### 8.1 图的边界

`program/graphs.*` 按 exact-B 准备 ordinary、control、target verify、draft chain 和决策图。
Resident Main 一次 replay 包含 embedding、全部层、PLE、final mixer，以及生成所需的 head。
Control 使用无 head 的图；前缀命中后的首次采样仍可单独从保存的 hidden 投影 head。
Eager/capture 调用同一组 GPU body，CPU experts、缓存 lease 与磁盘 I/O 在图外执行。

`text_graph.*` 将 resident 与 expert offload 的组织分开。Resident 的 QSA mixer 以 child
graph 持有各 context envelope；其余层计算直接捕获在外层图。PLE 消费也是 child，
包含 ready wait、计算和 consumed record。换上传槽只更新该 child，不复制完整 Main。
当前 Host producer 在 Main 提交前交付已排入 H2D 的 E；H2D 可以与图内前序计算重叠，
Host 读取等待不再与 resident 前序层重叠。表驻留位置不进入 Main 的数学接口。
Offload 仍按真实 CPU/cache 交接分段，PLE 之前的层可以在 Host 准备期间执行。

同一 exact-B 内，位置、valid columns、state selectors、table rows、page IDs、route/mask
内容是 Device 输入。RankProgram 选择 context envelope 并更新本地 child；Op 保留自己的
分派逻辑。启动时验证全部 envelope 更新和 PLE 槽切换；workspace 取允许 profile 的峰值。

MTP 按本轮实际深度 S=1..K 选择一条 draft chain，保留每行有效长度和 step-0 bridge。
每步在 transfer 分支异步复制 ID 并记录 external event，Host 按顺序 append 到 PLE；
链尾回接 compute stream。随后提交 Main verify（含 head），再提交消费 logits 的接受图。
Ordinary 同样将 Main/head 与采样图分开，约束 mask 和采样配置在决策前上传。
最终接受长度仍由 Host 事务决定，GDN fold、QSA/PLE tail 提取及 Backend 对齐在图外执行。

### 8.2 容量规划

加载前从 config、Binding/Use 和各 Op 容量查询建立执行 profile。
Startup 提供 resident 与目标可用的 offload 放置候选；Loader 按选定位置建立原编码 backing。
权重 materialize 后，Graph probe 使用少量私有 KV 页，并保留实际 batch、draft 数、
KV 格式、context envelope 与图共享关系。Resident 直接借用已加载 GPU banks；offload
使用公共 MoE calibration bank。Graph 准备描述区分两种执行方式。
销毁 probe 后按实际可用显存和 Graph 重建估计规划正式资源，遵循公共启动的 1 GiB
headroom 与有界容量纠偏规则。Graph 先于其引用的 workspace、bridge 和 events 销毁。

Flash Next 的 prefill chunk 默认 4096，可显式指定 128 的正整数倍，
实际窗口受剩余上下文、capture 点和媒体边界限制。没有启动自动 chunk 调优。
Resident 的自动 KV 使用扣除必需资源后的剩余预算；offload 先保证一个完整上下文，
剩余预算用于 expert cache。显式 KV 容量必须兑现。
Device pool 在启动时固定，不在请求期间扩容或动态与 expert cache 交换配额。

Workspace 对顺序互斥 scratch 取峰值，跨阶段存活的 records、conditioning 和视觉
handoff 单独计入。PLE 容量为
`Mmax=max(prefill_chunk,max_concurrency*(draft_tokens+1))`，
两份 pinned E 和两份 Device E 各自共占 `2*Mmax*2560*2` B，
另计 64 MiB row cache、I/O buffers 和有界 Host metadata。PLE 容量查询使用独立的
TableGeometry；启动重试不借用已销毁 ModelShard 的 table。Spark 将这些 Host 分配与 GPU
分配计入同一物理预算，Host State/KV cache 为零；Vision 的 live media 额度另在启动预算中预留。

以下是本实例的逻辑 state payload，尚未计入 field/slot 对齐：

| 内容 | 每个状态副本 |
|---|---:|
| Main GDN recurrent | 108 MiB |
| Main GDN conv history | 2.109375 MiB |
| PLE N history | 180 KiB |
| QSA raw tail + 块首坐标 | 780 B/层；Main 12 层，MTP 另 1 层 |
| Main sample / MTP conditioning | 5 KiB / 启用时另 20 KiB |

每个逻辑页组包含 pool 的全部层；下面按单层列出主 KV 加 4096 B index plane：

| 主 KV profile | 单层 64-token payload | Main 12 层 bytes/原 token |
|---|---:|---:|
| BFloat16（BF16 K / FP16 V） | 135168 B | 25344 |
| Int8Group64 | 71680 B | 13440 |
| Fp8E4M3Row256 | 70144 B | 13152 |
| Nvfp4Group16 | 40960 B | 7680 |
| Fp8KeyNvfp4Value | 55552 B | 10416 |

MTP pool 按一层另计，其容量包含 provisional lead。
实际分配从 expanded plane schema、StateImage layout 和 workspace 查询产生，
不以该 payload 表代替启动预算。完整 Host experts、registered 子集、pinned context、
PLE cache 和进程内存分别计费；registered 范围不是另一份专家权重。

## 9. 实现与维护入口

| 责任 | 实现 |
|---|---|
| 配置、只读定义与本地权重加载 | [config](../../src/models/qwen4_exp/config.cpp)、[definition](../../src/models/qwen4_exp/definition.cpp)、[load](../../src/models/qwen4_exp/load.cpp) |
| 无地址 profile、Parameters 与层计算 | [execution](../../src/models/qwen4_exp/execution/) |
| 启动预算、资源和图 | [planning](../../src/models/qwen4_exp/program/planning.cpp)、[resources](../../src/models/qwen4_exp/program/resources.cpp)、[graphs](../../src/models/qwen4_exp/program/graphs.cpp) |
| PLE hash、读取服务和上传槽 | [ple](../../src/models/qwen4_exp/ple/)、[PlePipeline](../../src/models/qwen4_exp/program/ple_pipeline.cpp) |
| Main/MTP 物理执行与采用 | [execution](../../src/models/qwen4_exp/program/execution.cpp)、[mtp](../../src/models/qwen4_exp/program/mtp.cpp)、[commit](../../src/models/qwen4_exp/program/commit.cpp) |
| 请求事务、页组与 checkpoint | [session](../../src/models/qwen4_exp/session/)、[directory](../../src/models/common/context/directory.h) |
| StateImage 与本地上下文执行 | [state](../../src/models/qwen4_exp/state/)、[context](../../src/models/qwen4_exp/program/context.cpp) |
| 静态 expert 排序 | [expert_priority](../../src/models/qwen4_exp/program/expert_priority.cpp) |

[Tests](../../tests/README.md)维护 Op 数学、状态语义、PLE 和模型加载测试入口；
[Benchmarks](../../bench/README.md)维护局部与完整 Engine 测量。
