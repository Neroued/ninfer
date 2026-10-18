# mHC 实现设计

性质：待落地的四流 residual Op 接口与融合设计。决策、实现状态和任务范围见
[设计索引](README.md)。完整数学、行序和与 Qwen GR 的区别见
[模型 mHC](../../maintainer/glm5next-model.md#2-main-与-mhc-residual)，精度与舍入边界见
[mHC 数值政策](../../maintainer/glm5next-quantization.md#5-mhc-数值边界)。

本文覆盖五个完整入口、native 布局、alias、workspace、相邻 site 融合和验收。
首个几何为 D=4096、S=4。`mhc` 与现有
[gated residual](../../../include/ninfer/ops/gated_residual.h) 保持不同计算合同；
可复用私有机制，不能通过模型名或 shape 改变现有 GR 数学。

## 1. 权重与 tensor 布局

Native Tensor 第零维连续。`T` 为每请求物理 token 宽度，`B` 为 batch，
`M=T*B`，`m=t+T*b`。数学按每列独立计算，不混合 token 或请求。

| 对象 | Native 布局 | 解释 |
|---|---|---|
| `residual / updated_residual` | BF16 `[4096,4,T,B]` | hidden 轴连续，随后是 stream |
| `embedding / block_input / block_output / hidden` | BF16 `[4096,T,B]` | 完整 block hidden |
| `post` | FP32 `[4,T,B]` | 已激活的 `q` |
| `mix` | FP32 `[4_new,4_old,T,B]` | `mix[j,i,t,b]=C_math[i,j,t,b]` |
| `F` | BF16，逻辑 `[24,16384]`，native `[16384,24]` | 每输出行的 K 轴连续 |
| `base / scale` | BF16 `[24]` / `[3]` | 与[模型系数行序](../../maintainer/glm5next-model.md#21-全宽-rms-与-24-行投影)一致 |
| `gamma / gamma_final` | BF16 `[4096]` | 直接 gain |

关键偏移以元素数表示：

```text
residual(d,i,t,b) = d + D*(i + 4*m)
block(d,t,b)      = d + D*m
post(j,t,b)       = j + 4*m
mix(j,i,t,b)      = j + 4*i + 16*m
F(a,k)           = k + 16384*a, k=i*D+d
```

源 `F` 行 `[8,24)` 的顺序为 `8+4*i+j`，与 native mix 的 `j` 连续轴相符。
Converter 按逻辑角色保值导入；不量化，不把 attention/FFN 不同输入的 F 合并为
一次共同投影，不在运行时重新 packing。

布局相同不意味着 GR 与 mHC 公式相同；现有 GR 合同和权重绑定保持独立。

## 2. 五个完整 Op

下列签名为语义记法，分号前为输入，分号后为输出；各计算入口另接调用方 workspace
与 `CudaExecutionView`。`W={F,base,scale,gamma}` 表示单个 site 的已绑定权重，
`P` 表示[数值政策](../../maintainer/glm5next-quantization.md#51-当前固定-profile)中的语义参数。实际 C++ 声明须完整表达这些输入输出，不暴露 kernel
分块、私有阶段或模型调度。

```text
mhc_expand_pre(embedding, W, P;
               residual, block_input, post, mix)

mhc_pre(residual, W, P;
        block_input, post, mix)

mhc_post(residual, block_output, post, mix;
         updated_residual)

mhc_post_pre(residual, block_output, post, mix, next_W, next_P;
             updated_residual, next_block_input, next_post, next_mix)

mhc_post_mean_norm(residual, block_output, post, mix, gamma_final, eps_final;
                   hidden)
```

### 2.1 入口效果

| 入口 | 完整效果 |
|---|---|
| `mhc_expand_pre` | 精确复制 embedding 为四流，再按当前 W 生成归一化分支输入和 `q/C` |
| `mhc_pre` | 消费完整 residual，执行系数生成、Sinkhorn、collapse 与分支 norm |
| `mhc_post` | 消费完整分支输出与旧 `q/C`，执行混合和 BF16 residual 更新 |
| `mhc_post_pre` | 当前 post 后，基于实际 BF16 新 residual 执行下一 site 的完整 pre |
| `mhc_post_mean_norm` | 最后一次 post、BF16 residual 边界、四流均值和 final norm；只输出 hidden |

`mhc_pre` 不输出 `p`；它只在本次调用内部使用。跨分支保存的是 `post/mix`，直到
对应 post 消费完毕。公共 post 接收已经激活／迭代的系数，直接按值应用；不重新
sigmoid，不重新 Sinkhorn，也不要求 C 的行列和严格为 1。

`post_pre` 的新旧 W、系数明确分开。旧系数只能完成旧 site 的 post，下一组系数
只能由新 residual 和 `next_W` 生成。相邻 site 的选择属于模型调用顺序。

### 2.2 有效域、写入和 alias

- `T` 为任意正物理宽度，受 views/storage 可表示范围约束；`B=1..8`。没有独立
  decode/verify/prefill 数学模式。没有有效列的执行不调用本 Op。
- 所有 tensor 连续、基址至少 16-byte 对齐，shape 必须匹配第 1 节。权重和数学输入
  使用有效模型数值；Op 不读取设备系数来做 Host 同步验证。
- 每个入口完整写入其全部输出列。Op 不接收有效长度或 padding mask；调用者为物理
  padding 列提供有效有限输入，在后续消费者处管理有效列。mHC 不发布历史状态。
- `mhc_post` 和 `mhc_post_pre` 的 `updated_residual` 可以与 `residual` 完全同 view，
  或完全不重叠；禁止部分 overlap。In-place 执行仍须按完整旧四流计算所有新值。
- 除上述特例，输出与所有输入、权重、其他输出互不重叠；新旧 `post/mix` 缓冲不能
  alias。被 in-place 修改的 residual 也不能与其他只读输入重叠。
- 只读输入在语义和形状允许时可以共享 backing；所有未声明可变的输入保持不变。
- `mhc_expand_pre` 的 residual 是精确 BF16 word copy；其他数学输出按统一数学
  精度验收，不因融合/独立执行而承诺全结果 bitwise 相同。

### 2.3 执行资源

每个入口使用 caller-owned、call-scoped workspace，与所有 operands 不重叠。
匹配的容量查询只读取 Host metadata，支持启动前的无地址 profile，覆盖查询范围内
所有合法 T/B 和完整调用的私有阶段、对齐需求；不能只报告某一个内部 kernel 的需求。
无需 workspace 的路线可报告零，不能借此缩小公共支持域。

入口向给定 execution stream 提交，支持固定地址下读取当前内容的 Graph capture/replay。
不分配隐藏设备内存，不做 Host readback，不发起通信，不拥有请求或长期状态。
精度政策、数学参数、别名规则不会因私有路线变化；kernel 数、tile、归约分解和
调度阈值属于实现，不作为新的公共模式。

## 3. 层序与融合组织

```text
embedding / 替换后的媒体特征
  → mhc_expand_pre(attention 0)
  → attention 0
  → mhc_post_pre(attention 0 → FFN 0)
  → FFN 0
  → mhc_post_pre(FFN 0 → attention 1)
  → ...
  → 最后一个 FFN
  → mhc_post_mean_norm
  → Main hidden → LM head / MTP conditioning
```

90 个 site 对应一次入口准备、89 个相邻 `post_pre` 和一次末尾收束。这是完整 Main
计算的逻辑 Op 组织，不承诺一次 Op 只发射一个 kernel。独立 `pre/post` 支持需要
独立边界的调用与数学验证；公共融合直接描述完整组合，不要求先调用这些独立 Op。

优化围绕以下数据依赖展开：

1. 全宽平方和与 24 行投影共享 residual 读取；不生成 16384 维 normalized tensor，
   不经普通 Linear 的 BF16 输出再生成 FP32 logits。
2. 小型 gate、4×4 softmax 和全部 Sinkhorn 步骤在片上连续完成，避免每轮显存往返。
3. post 形成 BF16 新 residual 后，尽量直接供下一次投影、统计和 collapse 使用，
   同时保留下一分支返回后仍要读取的 residual。
4. collapse 与其 norm 融合，避免单独物化低精度汇合向量；末尾同样融合 mean/norm。
5. 初始复制与第一组 pre 合并；末尾按合同在片上形成最终 BF16 四流而不必写回。

小 M 可评估融合 GEMV 或分块归约，大 M 可评估 BF16 操作数、FP32 累加的窄输出
GEMM 及后处理。路线选择必须按完整 Op 验证；不把 Sinkhorn 单步或 GEMM partial
变成模型可调用的公共 Op。

当前没有 kernel 或整链路性能结果。单 site、单 token 的系数投影是 393216 MAC，
collapse 是 16384 MAC，残差矩阵混合是 65536 MAC，注入另有 16384 次乘法。
这些只是几何计数，不包含 norm、非线性和迭代，也不代表实际延迟或融合收益。

## 4. 完整分支输出与执行资源

本地 Op 始终消费完整 BF16 y。KDA/FFN 局部贡献归约、MLA owner 输出汇集完成后，
两 rank 才各自执行 mHC；复制放置及通信交接由
[并行设计](parallel.md#1-首版并行分工)维护。本 family 不接收 rank 或 communicator。

ModelShard 拥有不可变权重，Parameters 借用绑定；RankProgram frame 拥有 residual、
block input/output、系数、workspace 和 Graph。Session 决定请求与列映射。
缓冲计费见[执行资源](parallel.md#4-mhc-执行资源)。mHC 激活不是历史状态，不进入
两个持久 pool，MTP 仅保留其实际需要的最终 Main hidden。

## 5. 验收与性能证据

### 5.1 独立数学 oracle

每个入口用独立 FP64 oracle 计算完整公式，解读公共输入实际表示的值；不通过生产
Op 串接或复刻 GPU staging 建立参考。固定 Sinkhorn 次数、矩阵方向、epsilon 位置
和 residual 表示边界必须直接写入 oracle。

数学出口的参考保留高精度结果用于误差比较；`post_pre` 和末尾融合中声明的 BF16
residual 边界单独表达，其他私有／公共最终输出不为匹配路线额外加入舍入步骤。
除完整融合公式检查，还要从实际返回的 BF16 updated residual 独立计算下一组
pre 参考，检查下一阶段确实消费该表示；不能仅看新 residual 误差而忽略系数来源。

初始化复制、未修改输入和界外区域用精确检查；浮点计算使用同一公共语义的误差
标准，不要求不同归约路线逐 bit 一致。误差标准在实现评估前按公共出口与数学确定，
不根据候选输出反向放宽。vLLM parity 和模型输出只提供补充证据。

### 5.2 必须覆盖的失效类型

| 场景 | 需要发现的问题 |
|---|---|
| 明显非对称 C、不同旧流输入 | `C[old,new]` 转置错误 |
| 四流幅值差异大 | 用 grouped RMS 代替全宽 RMS，或 collapse 读取 normalized X |
| 非平凡 pre/gamma | 多除以 4、归一化 pre、使用 offset gamma |
| 零输入、极端 logits、饱和 gate | epsilon 位置错误、softmax 溢出、错误裁剪 |
| 不平衡 mixing logits | 迭代次数／行列顺序错误，错误依赖严格双随机性 |
| residual 位于 BF16 舍入边界附近 | fused pre 使用未舍入新 residual |
| collapse/均值存在明显舍入差异 | 插入额外 BF16 中间边界 |
| 最后 post＋mean＋norm | 跳过最后一次注入/混合、漏 norm 或重复 norm |
| 多 token、多 batch、列重排 | 串流、串列、跨请求混合 |
| exact in-place 与 disjoint 输出 | 旧四流未读完就被覆盖、partial overlap 未拒绝 |
| Graph 重播且 bound address 内容变化 | 缓存旧输入或系数、隐藏分配与 Host 同步 |

覆盖五个公共入口及所有可达私有路线；T 包括 1、路线边界两侧和代表性内部值，B
覆盖 1..8 的代表值及上界。融合与独立路线分别对数学 oracle 验收，而非仅彼此对齐。
后续跨 rank 接入还需检查完整 y 只被消费一次，以及双方输入 words 和行序一致。

### 5.3 性能范围

以 Spark 上相同几何、精度和数学的独立 pre/post 组合为融合对照，覆盖小 batch
decode、多 token verify 与大窗口 prefill；比较完整 `pre`、`post_pre`、末尾收束及
组合链路。明确硬件、工具链、T/B、Graph/cold 条件、workspace、分布与最差变化。

单独 Sinkhorn 或 GEMM 的加速不能代替整个 Op 的结果；本地 Op 结果不能声称为
双机模型端到端收益。跨 rank 的性能评估必须把输出归约／汇集、pack/unpack 和后续
消费者的交接算在内。当前仅有结构和容量推导，未实现、未执行性能比较。

## 6. 接管与退出条件

数学与数值来源分别在长期模型／数值文档维护。相邻 post/pre 组织参照本地 vLLM
revision `00b7847c8036b667742b4efb21aab1de51fd4721` 的
[GLM 调用层序](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/model.py)；
后端舍入差异按数值政策处理，不能直接作为 NInfer 的 oracle。

五个入口及测试落地后，实际 Op 头文件与测试接管公式组合、alias、写域和精度验收；
模型接入后，执行代码与当前执行说明接管层序和资源。未实现部分仍保留在本文，
全部接管后删除设计，不在 maintainer 目录保留另一份待实现接口规格。
