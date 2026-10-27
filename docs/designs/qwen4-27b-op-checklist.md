# Qwen4-27B：RTX 5090 Op 需求清单

日期：2026-10-27。

本清单以[架构与量化方案](qwen4-27b-rtx5090.md)为固定输入，准备单张 RTX 5090
的生产 Op 支持与优化。配置、recipe、精度选择理由和权重预算由该方案维护；
本清单维护具体几何、支持缺口、实施顺序和验收条件。

范围包括 Text、原生 MTP 及现有 Vision 的复用，不包括 Spark、TP、多卡通信、
MoE 或 PLE。源码状态来自本次合同与生产分派检查；“已有”不表示本次已完成 GPU
数值或性能测量。下列复选项表示后续实施工作，当前未标记任何新增 Op 完成。

## 1. 共同约定

- 权重写作 `W[N,K]`，N 为输出行，K 为输入特征。普通投影消费 BF16 `[K,M]`，
  产生 BF16 `[N,M]`；`M=T*B`，列号 `m=t+T*b`。
- HC 的四条 streams 是特征轴。只有 MTP `fc_hidden` 将它们作为四倍 Linear 列，
  调用列数为 `4M`；其他 Text 投影仍为 M 列。
- 格式和 activation policy 直接遵循方案。特别是 Main/MTP Indexer、普通 HC 和
  final HC down/up 均为 FP8/AllowA8；HC injection 为 BF16/A16Only。
  `AllowA8/AllowA4` 是许可，小 M 可使用 A16，不按 Main/MTP 身份选择不同数学路径。
- 一个待办由 Op、格式、几何、分段/epilogue 和计算许可共同确定。纯 Linear 的
  shape 已注册，不代表 LinearSplit、SwiGLU 或 HC 已支持；不要求为每个融合 Op
  再新增一个无调用方的纯 Linear 入口。
- 数学、资源和验证规则引用 [Op 开发](../maintainer/op-development.md)；
  Linear 调优范围引用 [Linear tuning](../maintainer/linear-tuning.md)。
  权重物化属于准备阶段，Op 借用权重与 caller-owned workspace，执行中不隐藏分配或重打包。

## 2. 首批新增矩阵计算

| 编号 | 生产 Op / 用途 | W[N,K] | 格式 / policy | 当前缺口 |
|---|---|---|---|---|
| L1 | `linear_swiglu`，Main/MTP Dense FFN gate/up | `[28672,5120]` | NVFP4 / AllowA4 | 新增几何；现有 NVFP4 主干 SwiGLU 为 `[34816,5120]` |
| L2 | `linear`，Main/MTP FFN down | `[5120,14336]` | NVFP4 / AllowA4 | 新增几何 |
| L3 | `linear_split`，Main/MTP QSA 完整输入投影 | `[14976,5120]` → `[1024,1024,128,6144,6144,512]` | FP8 / AllowA8 | 新增六段合并投影；顺序为 K/V/index-K/Q/gate/index-Q |
| L4 | `linear`，MTP `fc_embedding` / `fc_hidden` | `[5120,5120]` | FP8 / AllowA8 | 新增一个几何，分别处理 M / 4M 列 |
| L5 | `linear`，MTP optimized draft head | `[248320,5120]` | Q4 G64 / A16Only | 新增几何；现有同词表 Q4 head 的 K=2560 |

- [ ] L1：gate/up 行序为 `[gate14336;up14336]`，输出 BF16 `[14336,M]`。
  验证完整 SwiGLU 数学及所选 A16/A4 路径，不以独立投影先舍入 BF16 的结果替代融合参考。
  NVFP4 合并权重必须保持源 scale/divisor 语义；不兼容的元数据不能直接共用一个 divisor。
- [ ] L2：输出独立 BF16 `[5120,M]`，交给 HC combine；不使用普通 `linear_add`
  将结果加到单路 hidden，HC 的回注由 HC Op 完成。
- [ ] L3：按[合并行序](qwen4-27b-rtx5090.md#121-合并权重与行视图)一次调用直接
  产生六个连续 BF16 输出，Main/MTP 共用相同 profile。实际 A8 路径共享一次输入
  编码，不将六段包装为多次独立 Linear/Indexer 投影；分段、生产分派与完整容量查询
  同时补齐。具体激活 view 和消费阶段见第 4 节。
- [ ] L4：同一权重几何覆盖 stem 的两次调用。`fc_hidden` 权重仍是 `[5120,5120]`，
  不能错误扩大为 `[20480,20480]`；重点覆盖 4M 的实际列数。
- [ ] L5：输出完整词表 logits，行号就是 token ID。复用现有草拟 argmax 流程，
  不增加词表裁剪、token remap 或新的 head 格式。

实现沿用 `src/ops/linear/{projection,split,swiglu}/` 及其共享 kernels。
生产分派、构建注册、容量查询、数值用例和对应 benchmark 的可调用性一起完成。

### 2.1 可直接复用的矩阵入口

| 编号 | 生产 Op / 用途 | W[N,K] 与分段 | 格式 / policy |
|---|---|---|---|
| R1 | `linear_split`，GDN QKVZ | `[16384,5120]` → `[10240,6144]`，QKV/Z | NVFP4 / AllowA4 |
| R2 | `gdn_control_proj`，GDN A/B | `[96,5120]`，A48/B48 | BF16 / A16Only，输出 FP32 g/beta |
| R3 | `linear`，GDN output 与 QSA O | `[5120,6144]` | NVFP4 / AllowA4 |
| R4 | `linear`，Main 与共享 full MTP head | `[248320,5120]` | FP8 / A16Only |
| R5 | `embedding`，Main/MTP 共享查表 | `[248320,5120]` | FP8，输出 BF16 |

现有分派依据：[LinearSplit](../../src/ops/linear/split/dispatch.cpp)、
[NVFP4 Linear](../../src/ops/linear/projection/nvfp4/nvfp4_dispatch.cpp)、
[FP8 Linear](../../src/ops/linear/projection/fp8/fp8_dispatch.cpp)、
[GDN controls](../../include/ninfer/ops/gdn_control.h)、
[Embedding](../../src/ops/embedding/embedding.cu)。

QSA 整投影统一采用 L3，历史补齐采用第 4.3 节 M1 的连续前缀三段投影。
这是本配置的正式布局与调用需求；各角色 Binding 用于逻辑映射，不分别发起投影。
共享的量化、GEMM、分块或归约 kernel 由 Op 内部组织，不额外维护按 Main/MTP
身份拆分主投影与 Indexer 的执行路径。

已有入口先复用。只有本次变更影响其行为、现有证据不足或测量暴露实际瓶颈时，
才增加相应回归或优化，不重做所有已有 shape 的 benchmark。

## 3. HC：几何与混合格式合同

目标 H=5120、streams=4、residual width=20480、lowrank=640。
当前 [HC 合同](../../include/ninfer/ops/gated_residual.h)和
[实现几何](../../src/ops/gated_residual/operands.h)限定 H=2560、rank=320，
mix 的 down/injection/up 都是 BF16。目标需要以下扩展。

| 编号 | Op / 计算部分 | 目标输入、权重或输出 | 工作 |
|---|---|---|---|
| H1 | `hc_expand_norm`、`hc_grouped_rmsnorm` | embedding `[5120,M]`；residual/normalized `[5120,4,M]`；norm `[5120,4]` | 扩展 H=5120 |
| H2 | 普通 `hc_mix` down/up | `[640,20480]`、`[20480,640]`，FP8/AllowA8 | 扩展几何、权重表示、activation policy 与实现 |
| H3 | 普通 `hc_mix` injection | 独立 BF16 `[4,20480]`；输出 FP32 `[4,M]` | 将 injection 与 FP8 down/up 的表示分开表达 |
| H4 | final `hc_mix` | H2 的 down/up，无 injection；输出 `[5120,M]` | Main/MTP 统一 FP8/AllowA8 |
| H5 | `hc_combine`、`hc_combine_norm` | residual `[5120,4,M]`、block `[5120,M]`、injection `[4,M]` | 扩展 H=5120，包含 MTP stem 使用的系数为 1 的广播相加 |

- [ ] H1/H5：每条 stream 独立 offset RMSNorm，gain=`1+w`；combine 先产生实际
  BF16 residual，下一次 norm 消费这些 BF16 words。覆盖允许的原地更新与不重叠输出。
- [ ] H2/H3：完善 HC 所属的权重/profile 与容量接口，明确 FP8 down/up 和 BF16
  injection 各自的策略。保留 `/4`、SiLU、sigmoid、stream mean 与 FP32 injection
  的完整语义，不把 mixed-format 判断或 HC 私有舍入边界移交模型调用方。
- [ ] H2/H4：支持 K=20480 和 K=640 的实际 FP8 激活编码与计算；当前
  [A8 编码实现](../../src/ops/linear/kernels/fp8/fp8_a8.cu)只覆盖部分 K，不能将
  “已有 FP8 kernel”当作这些几何已经可运行。处理 K=640 的调度/向量尾部。
- [ ] H4：final mixer 无 injection，也不额外添加 final RMSNorm。
  Main/MTP 使用同一精度合同；为 MTP 保留的是 mixer 前的 residual。
- [ ] 在完整 HC Op 上验证数学和性能。不得仅凭两个纯 Linear 的测试宣布 HC 完成，
  也不要求先将 HC 私有中间量变成额外的公共 BF16 张量。

精度风险按方案第 3.1 节处理：尤其区分逐特征 mixing 与广播整条 stream 的 injection。
H2/H3 的融合可以由实现选择，保留 BF16 injection 不意味着必须独立启动一个 kernel。

## 4. QSA 与 MTP 历史补齐

### 4.1 QSA Op

| 编号 | Op | 目标 | 当前状态 |
|---|---|---|---|
| A1 | `qsa_prepare` | 24Q/4KV、D=256；Indexer 4Q/1K、D=128；rotary=64；FP8-row256 KV | 扩展；当前 prepare 限制 KV heads=1/2，并按 ratio=12 解释 Q heads |
| A2 | `qsa_prepare_kv` | 4 KV heads，主 KV 与 compressed index-key 写入 | 扩展，供历史补齐使用 |
| A3 | `qsa_sparse_gqa` | 24Q/4KV，GQA ratio=6，FP8-row256 cache，sigmoid gate | 新增几何与对应生产执行路径；当前为 24/2、12/1 |
| A4 | `qsa_indexer` | 四-token 块，最多选 512 块；最大历史 262144 tokens | 已有，复用 |
| A5 | `qsa_tail_extract` | raw index-K tail、块首坐标、实际接受前缀 | 已有，复用 |

- [ ] A1/A2/A3：同步 [QSA 合同](../../include/ninfer/ops/qsa.h)、验证、workspace
  和 RTX 5090 分派；实现 ratio=6 的 head 映射及完整四头 KV 读写，不能只放宽参数检查。
- [ ] 保留 Main K/V 的完整历史。四倍压缩仅作用于 index key；raw key 四列平均后
  按合同舍入，再 norm/RoPE，使用实际块首 MRoPE 坐标。
- [ ] Sparse GQA 保留 A16 attention 计算、片上 FP8 KV 解码和 gate 前的完整归约。
  Indexer score 不加入主 attention logits；选中完整块之外还读取因果可见的未完成尾块。
- [ ] 覆盖跨窗口/页边界、0–3 列 tail、零有效列、混合 batch 长度和 Graph replay。
  在 C=511/512/513 个完整块验证全选与 TopK 转换；最长容量验证到 262144 tokens。

现有实现入口：[prepare](../../src/ops/softmax_attention/qsa/prepare/prepare.cpp)、
[sparse GQA](../../src/ops/softmax_attention/qsa/sparse_gqa/sparse_gqa.cpp)。

### 4.2 各阶段的 view 与调用

权重 parent、行区间及 scale 映射以[方案第 1.2.1 节](qwen4-27b-rtx5090.md#121-合并权重与行视图)
为准。L3 与 M1 均消费本次 HC 输出的 BF16 `[5120,M]`，每次只调用一次对应的
`linear_split`。View 选择由模型阶段决定，Op 自身只接收明确的权重、分段和输入几何。

| 阶段 | 权重 view | 投影输出与下游消费 |
|---|---|---|
| Main prefill、Replay、scoring | 本层完整 `[14976,5120]` | L3 六段 → A1 prepare → A4 indexer → A3 sparse GQA → O；需要每列的完整 attention |
| Main ordinary decode、单列 control | 本层完整 `[14976,5120]`，T=1、M=B | 同一六段入口，使用本次 batch 的位置与有效列 metadata |
| Main target verification | 本层完整 `[14976,5120]`，T=d+1、M=T*B | 六段一次投影；A1 写当次 raw/坐标 records 和候选 KV，各 query 只读自己的因果历史 |
| MTP 每一步 draft | MTP 层完整 `[14976,5120]`，T=1、M=B | 六段 → 完整 QSA；各步使用其实际 Main/MTP conditioning，沿用现有 bridge tail 合同 |
| MTP prefill、Replay、control 的历史建立 | MTP parent 前缀 `[2176,5120]` | M1 三段 → A2 prepare-kv → A5 tail extract；不执行 query/indexer/readout/O |
| 接受后的 MTP 历史重建 | 同一 MTP 前缀 `[2176,5120]` | 从 bridge tail 接续，用 Main conditioning 与已接受 shifted IDs 重建所需历史；一次三段投影覆盖本窗口 |
| Main 接受前缀的 tail 提取 | 无投影权重 view | A5 读取原 verification records 与原基准 tail；不缩短 T 重新投影 |
| 完整 State/KV snapshot 恢复或前缀命中 | 无投影权重 view | 恢复已有状态；需要 Replay 的未覆盖后缀按上述对应阶段执行 |

历史窗口的 T/M 取实际配对后的输入列数：根部跳过没有前驱 conditioning 的首列，
连续 chunk 接上旧 conditioning；接受后的重建沿用真实 bridge，仅补后续所需列。
没有历史工作时不提交空 Linear。历史补齐不能直接采用后续 draft 的投影结果，因为
它们可能消费 MTP conditioning，而需要重建的是以 Main conditioning 配对的历史。
配对与前沿由[模型执行合同](../maintainer/qwen4_exp-execution.md#5-mtp-的生成与采用)负责。

### 4.3 MTP 三段前缀投影与权重生命周期

| 编号 | Op / 用途 | 逻辑权重 view 与输出分段 | 格式 / policy | 当前缺口 |
|---|---|---|---|---|
| M1 | `linear_split`，MTP 历史 K/V/index-K | `[2176,5120]` → `[1024,1024,128]` | FP8 / AllowA8 | 新增三段合并投影及完整 parent 的前缀 view 支持 |

- [ ] M1：一次投影只计算前三段，不为历史构造 Q/gate/index-Q，也不拆成 K、V、
  index-K 三次 Linear。A8 路径共享本次 `[5120,M]` 的输入编码，输出直接交 A2。
- [ ] 支持 L3 完整 view 与 M1 只读前缀 view，正确表达逻辑 shape、codes/scales
  可读范围和完整 parent backing。当前 [native weight](../../src/core/weight_view.cpp)
  要求 row-scaled FP8 使用完整 parent，[FP8 operands](../../src/ops/common/host/operands.cpp)
  也验证紧凑 parent 布局；单独注册 shape 不能满足需求。前缀 scale 指针使用 parent
  的真实 scale plane，不以 `2176*5120` 作为新的 scale-plane offset。
- [ ] 转换与准备阶段完成合并、行重排和绑定；参数所有者为完整投影、历史投影建立
  共享 backing 的稳定只读 views，保证覆盖所有 eager 调用与 Graph replay 的生命周期。
  执行阶段不复制或重打包权重，不持久保存第二份 history 权重。
- [ ] 准备、执行分派与容量查询使用同一组 view/分段合同。分别覆盖完整六段和前缀
  三段的实际 M 区间及 scratch；不能沿用旧两次投影的 workspace 估算。
- [ ] 覆盖 A16 与实际启用的 A8 路径。完整与前缀使用相同逻辑权重和编码语义，
  不要求不同 M、不同 reduction schedule 的输出逐 bit 相同。

### 4.4 激活、缓存与 records 的 view

合并的是权重和投影调用。`linear_split` 直接写到互不重叠、16-byte aligned 的
连续 BF16 分段输出；这些输出可以来自同一 arena 的不同区域。

| 逻辑输出 | LinearSplit 输出 view | 下游 QSA view | 最后消费方 |
|---|---|---|---|
| K / V | 各 `[1024,M]` | 各 `[256,4,T,B]` | A1 或 A2 的 KV 编码/写入 |
| raw index-K | `[128,M]` | `[128,T,B]` | A1 或 A2 的压缩与 raw records 写入 |
| Q | `[6144,M]` | `[256,24,T,B]` | A1 norm/RoPE 后交 A3 |
| output gate | `[6144,M]` | `[256,24,T,B]` | A3 的 sigmoid gate |
| index-Q | `[512,M]` | `[128,4,T,B]` | A1 norm/RoPE 后交 A4 |

- [ ] 分段输出按 `m=t+T*b` 零拷贝 reshape，保留实际 tensor strides。不能先写
  `[14976,M]` 或 `[2176,M]` 大矩阵，再将行切片误当成连续 `[N_s,M]`：M>1 时
  其列跨度仍是 parent 的总行数。生产路径直接分段写出，不追加搬运/重排 kernel。
- [ ] Q、index-Q 可按 A1 合同分别原地 norm/RoPE；gate 保留原始 logits 到 A3。
  三段历史路径只分配 K/V/raw index-K 及必要 records，不借用虚构的 query 输出。
  Scratch 和分段 buffer 在最后一次设备消费结束前保持有效。
- [ ] A1/A2 将 raw index-K 的 BF16 words 与实际三轴坐标写入独立 records：
  `[128,R,B]` 与 `[3,R,B]`，沿用实际 offsets、valid columns 和 B 行身份。
  这些 records 保留到对应 tail 提取/采用完成，不借用会被下一次投影覆盖的瞬时输出。
- [ ] 每层主 K/V 的 FP8 codes/FP16 row scales 使用四 KV heads 的 paged views；
  compressed index keys 使用 BF16 `[128,16,1]` per 64-token record，沿用该层 KV
  的 page 映射。Main/MTP 各自的 cache 与 tail backing 独立，合并投影不合并状态。
- [ ] `valid_columns` 只限制 prepare、cache、records 与 attention 的有效访问，
  不按每行有效长度拆分已合并的投影调用。固定 T/B 的 Graph 使用稳定 views 和
  当前 Device metadata；RankProgram 为所有物理输入列提供有限安全值，padding 列
  可以参与投影但不写入有效 cache/records。接受前缀从当次 records 提取，未接受后缀
  不发布为有效历史。

## 5. GDN 状态链与基础 Op 的复用

GDN 目标几何沿用已有 Q/K 16 heads、V 48 heads、Dk=Dv=128，
conv channels=10240、4 taps、FP32 recurrent state。

| 入口 | 用途与保留语义 |
|---|---|
| R1 + `causal_conv1d_silu_split` | Prefill/Replay 输入；QKV 经过卷积/SiLU，Z 绕过卷积 |
| `gdn_input_proj_conv_update` / `snapshot` / `record` | NVFP4 `[16384,5120]` / AllowA4；覆盖单列更新、候选 snapshot 与 verification records |
| R2 `gdn_control_proj` | 由原输入计算 A/B 与 FP32 g/beta，不另加输入 norm 或公共 BF16 controls 投影边界 |
| `gated_delta_net` / `gated_delta_net_batch_update` | Q/K normalization、delta-rule 与 FP32 state 更新 |
| `gated_delta_net_replay_record` / `GdnReplayFoldPlan` | 保存实际消费的 records，按接受前缀恢复 state |
| `sigmoid_gated_rmsnorm` | D=128、48 V heads；共享直接 gamma，输出 sigmoid gate |
| `rmsnorm` | MTP pre-FC embedding D=5120、hidden D=20480；后者沿整个 residual 向量归约 |

- [ ] 接入时复用这些入口。受影响的状态路径验证 record/fold 与同一次物理 block
  snapshot 对齐；不缩短 T 重跑投影来模拟接受前缀。
- [ ] GDN 输出使用 sigmoid gate，不能误用已有 SiLU-gated norm；MTP hidden norm
  不能替换成 HC 的分 stream norm。

合同入口：[GDN input](../../include/ninfer/ops/gdn_input_proj.h)、
[GDN](../../include/ninfer/ops/gated_delta_net.h)、
[Replay fold](../../include/ninfer/ops/gdn_replay.h)、
[RMSNorm](../../include/ninfer/ops/rmsnorm.h)。

## 6. Vision：直接复用完整实现

Vision 的格式和计算策略沿用方案第 4 节；本节只列复用几何，不安排格式替换或新实现。

| 用途 | W[N,K] | 现有格式 / policy |
|---|---|---|
| Patch projection | `[1152,1536]` | Q6 / A16Only |
| QKV projection | `[3456,1152]` | Q4 / A16Only |
| Attention output | `[1152,1152]` | Q5 / A16Only |
| MLP fc1 | `[4304,1152]` | Q4 / A16Only |
| MLP fc2 | `[1152,4304]` | Q5 / A16Only |
| Merger fc1 | `[4608,4608]` | Q8 / A16Only |
| Merger fc2 | `[5120,4608]` | Q8 / A16Only |

复用[公共 Qwen Vision](../../src/models/common/qwen/vision/)的 bias、LayerNorm、
GELU-tanh、16 heads × D72 非因果 attention、二维 RoPE、position embedding 和
spatial merge。模型接入时验证 merger 输出与 Text 5120 维输入交接；不重复开展
未受影响 Vision Op 的实现或调优。

## 7. 验收与实施顺序

### 7.1 Op 完成条件

通用要求执行 [Op 开发](../maintainer/op-development.md) 和
[Linear tuning](../maintainer/linear-tuning.md)，本任务补充以下实际负载与风险点：

- Linear 类热区覆盖实际调用的 M=1…128，优先看 1、4、8；大 M 分别测 512、1024。
  MTP draft 使用 M=B，现有 draft 数 d=1…5 的 target verification 使用
  `M=(d+1)*B`，B=1…8；`fc_hidden` 按其真实 4M 列数测量。
  这些是调优负载，不是新的支持范围上限。
- 状态 Op 保留 T/B 两个轴，按各自合同选择有效范围；不把 Linear 的扁平 M
  范围机械套到 GDN/QSA。QSA 另覆盖历史长度、选块数和混合有效前缀。
- 独立数学参考消费实际存储权重、激活量化和公开舍入边界；测试实际生产分派，
  不能用两个实现互相一致代替数值正确性。尤其验证 HC injection、GDN controls、
  FP8 row scales、NVFP4 divisors 和 SwiGLU 非线性。
- QSA 使用有区分度的逐行 scales 验证六段行重排，覆盖源 Q/gate 解交错和 index-K
  前置；三段前缀从完整 parent 借用真实 scale plane，读取范围仅限前三段。M>1、
  B>1 的用例检查分段输出与 T/B reshape，避免 M=1 掩盖错误列跨度。历史建立、
  接受后的 MTP 重建和 Main records 提取分别按第 4.2 节覆盖。
- 公共容量查询覆盖所声明区间的实际 scratch，包括激活编码与归约；检查必要的
  不重叠、原地更新和状态写入域。Graph replay 使用当前绑定地址中的输入与 metadata。
- 性能结果明确实际 A16/A8/A4 路径、完整 Op 计时边界及 workspace 成本。与改动前
  同格式、同精度、同几何和同负载比较；新增 shape 单列为新增能力，不虚报已有优化收益。

测试与基准优先扩展现有 family：

| 范围 | 测试入口 | 基准入口 |
|---|---|---|
| Linear / SwiGLU / Split | `tests/ops/linear/{projection,swiglu,split}/` | `bench/ops/linear_bench.cu`、`bench/ops/nvfp4_linear_swiglu_bench.cu`、`bench/ops/linear/split/` |
| HC | `tests/ops/gated_residual/` | `bench/ops/gated_residual/` |
| QSA | `tests/ops/softmax_attention/qsa/` | `bench/ops/softmax_attention/qsa/` |
| GDN | `tests/ops/linear_attention/gated_delta_net/` | `bench/ops/gdn_input_proj/`、`bench/ops/gdn_control/`、`bench/ops/linear_attention/gated_delta_net/` |
| Embedding / norm | `tests/ops/embedding/`、`tests/ops/normalization/` | `bench/ops/embedding_bench.cu`、`bench/ops/rmsnorm/` |

### 7.2 正式权重后的精度验收

Op 正确性可以先用合成权重完成；模型量化质量需要正式权重与真实输入。
这两类完成状态分别记录，未测量的质量或端到端性能不标为已通过。

- Main Indexer：长上下文下检查选块变化及检索质量；C≤512 的全选不构成量化
  选择精度证据，也不能仅以块集合完全一致作为质量标准。
- HC：检查多层 residual 和最终输出；局部矩阵误差不能单独证明广播 injection
  或低秩 mixing 的模型质量。
- MTP：固定 Main、请求负载、采样和 draft 数，对比原 A16Only 限制与统一
  AllowA8 后实际运行的路径；一起报告接受长度、每轮耗时、最终生成速度。
  若两者都选择 A16，应明确没有获得 A8 数值或性能证据。
- 首批维持既定权重格式；方案第 5.2 节的 MTP FP8 升精度只在测量支持后另行采用，
  不作为本清单的预先依赖。

### 7.3 实施顺序

1. **矩阵准备**：L1/L2/L3，复用 R1–R5；先完成生产可调用性与数值验证，再调优目标 M。
2. **新架构关键 Op**：H1–H5、A1–A3；复用 A4/A5 与 GDN 状态链。
3. **MTP 补齐**：L4/L5、M1 三段合并投影及 FP8 前缀 view 依赖；沿用统一计算许可。
4. **接入验证**：复用 Vision、Embedding、norm 和现有状态机制；正式权重到位后
   完成第 7.2 节的模型质量与端到端验收。

本清单完成不等于模型接入完成。现有 Qwen4Exp config/执行使用 MoE 且要求 PLE，
后续模型工作仍需在同一架构入口表达 Dense FFN、无 PLE、对应绑定和执行规划；
这些是模型层职责，不通过新增按 checkpoint 命名的 Op 或执行路径解决。
