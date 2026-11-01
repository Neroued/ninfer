# 权重表示与加载

`.ninfer` 保存模型配置、编码对象、逻辑绑定、使用许可及资源。原生模型定义层连接与状态数学，
artifact 不保存一套可执行模型图。本文记录转换、存储与加载的分工；字段和 codec 的精确合同由代码维护。

## 从源权重到执行参数

```text
源 checkpoint → 模型 adapter / recipe → 编码对象与 bindings / uses
             → artifact reader / binder → 并行选择与物化 → ModelShard → Op 参数视图
```

| 部分 | 职责 |
|---|---|
| Adapter / recipe | 源角色映射、精简 config、量化方法、合并矩阵、保值行序与资源选择 |
| Artifact reader | 文件 framing、对象范围与资源读取 |
| 模型 binder | 所选组件的逻辑参数、shape、Use 关系和资源语义 |
| ParallelPlan | 本 rank 的逻辑切片、复制、合并与 expert 放置 |
| Materializer | 按选择读取编码、建立稳定 backing、传输与位置归属 |
| ModelShard / Parameters | 持有不可变本地权重 / 借用实际表示与计算参数 |
| Op | 消费数学视图，检查本实现的表示、几何和执行要求 |

源 checkpoint 的名称、来源文件和量化元数据不决定模型执行注册。
同架构的不同表示通过 bindings 与 Uses 接入；同一个物理对象可以供多个逻辑角色共享。

## 对象、绑定与使用

V3 的一个入口和可选续卷组成有序逻辑 payload。对象偏移属于这个地址空间，物理文件边界
不改变对象的 codec 或身份。Reader 通过目录定位文件，调用者不从续卷名称猜参数位置。

Binding 描述逻辑参数与对象区域的关系，Use 描述它在某个输入位置的 activation permission
和辅助值。同一权重可有不同用途；weight divisor 属于权重表示，activation divisor 属于使用位置。
资源是带明确消费者的 bytes，可承载 tokenizer/template，也可承载 PLE hash controls。

可选组件只绑定和物化实际选择的依赖。Main head 与 optimized draft head 可以有不同表示；
full proposal 保留 token ID 行序，indexed proposal 额外持有行映射，均不改变 target 词表。

文件与目录由 [schema](../../src/artifact/schema.h)、[framing](../../src/artifact/framing.h)
和 [Python writer](../../tools/artifact/)维护；模型角色由各架构 load 与 converter 维护。

## 数值表示与物理布局

Format 定义 codes、scales 与重建值；layout 定义这些 words 的 packing、planes 和 padding；
转换方法定义如何从源值产生编码。这三者与 Op 私有计算精度分别管理。

| 表示 | 需要保留的数值关系 |
|---|---|
| Groupwise INT | 有符号 code 乘实际存储的 FP16 group scale；group 轴与合法 code 域由格式固定 |
| NVFP4 | E2M1 code × E4M3 block scale ÷ FP32 matrix divisor；每个矩阵有自己的 divisor 域 |
| Row / block FP8 | E4M3 code 乘对应行或块的 BF16 scale |
| NF4 | 固定 codebook、E4M3 group scale 与 divisor；PLE 的 BF16 lookup 有精确舍入要求 |

NVFP4 源若使用 multiplier，导入为 divisor 会产生 FP32 reciprocal 舍入；保留 codes/scales
不等于承诺所有重建值与源逐位相同。量化 encoder 使用实际存储 scale 的规则、零值、尾部和
舍入顺序由对应实现及独立数值测试维护，不在模型文档复制算法。

`A16Only`、`AllowA8`、`AllowA4` 表达允许的 activation 表示集合。融合共享量化取相关 Use
的交集；许可不承诺实际执行某种 MMA。验证标准按公开许可选择，见[Op 开发](op-development.md#数值验证)。

权重格式与运行时 KV codec 分开。KV 的精确编码由
[kv_cache_format.h](../../include/ninfer/kv_cache_format.h)维护；它不改变源权重。
格式 registry 和布局寻址分别见 [formats](../../src/artifact/formats.cpp)、
[layouts](../../src/artifact/layouts.cpp)和 [Weight views](../../src/core/weight_view.h)。

## 合并与分片物化

合并投影按共同输入、兼容表示与元数据组织，逻辑角色仍有独立 Binding。
优先将 history-only 计算所需行放成连续前缀，让完整投影和历史补齐直接借用同一 parent。
LinearSplit 负责输出映射；它不决定模型的参数合并或并行切分。

分片加载从逻辑参数生成矩形编码选择，直接构造本地权重，不先把完整模型放进 GPU 再裁切。
[Encoded slice](../../src/artifact/encoded_slice.h)可以重排 scale tiles、拼接局部 GU、
保持 divisor 域，并在必要时 refine block-FP8 scales：细块重复原 scale words，
codes 与 represented values 不变。它是加载期的保值物化，不是再次量化。

普通零拷贝 view 继续保留原 parent 的寻址关系；不能把非连续 planes 伪装成连续 payload。
不能保值表达的切片由所属格式处理或拒绝。模型声明逻辑选择，不在模型里另写 codec；
执行期只消费准备好的稳定视图，不隐藏 packing 或设备分配。

共享分片实现见 [parallel_weights](../../src/models/common/parallel_weights.cpp)。
具体量化取舍与特殊连续前缀见 [Qwen4Exp](qwen4_exp-model.md)和[GLM5Next](glm5next-model.md)。

## Backing 与传输

位置是运行时事实，不写入持久格式。ModelShard 持有 Device、普通 Host、registered Host
或文件 backing；借用视图不接管其寿命。注册只提供 CUDA 可访问的传输视图，CPU 数学继续使用
Host 地址。先结束消费者和注册，再释放来源。

读取按真实文件对齐与选择范围规划，使用有界 staging；同一来源的多个目的可以共享读取。
对象跨文件不改变数值布局。元数据校验、语义绑定和 Op 支持各由所属模块报告，
加载路径不为已由可信 producer 建立的每个数值不变量扫描全模型。

转换命令见[用户指南](../weight-conversion.md)，代码入口为
[converter](../../tools/convert/)、[materializer](../../src/artifact/materializer.h)和
[backing](../../src/artifact/backing.h)。
