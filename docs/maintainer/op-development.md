# Op 开发规范

Op 是显式输入上的数学与局部状态合同。Kernel 是它的一种实现或一个内部阶段。
本文保留跨 family 的设计和验证方法；具体接口以 `include/ninfer/ops/` 为准。

## 语义边界

Op 完整定义输入、输出和允许修改的状态，独立于模型调用次序。模型拥有层连接、请求前沿、
接受长度、checkpoint 和通信交接；Op 不接收这些策略对象来推断自己该做什么。
纯地址与字节数的复制属于 Core，带索引、cast、transpose 或 scatter 数学的操作属于相应 Op。

Family 按计算与维护关系组织。相同模型层的多个计算不自动成为一个 family；
只有一个调用者、一个 shape 或一个优化 target 也不妨碍形成语义完整的 Op。
融合入口定义完整组合和可观察表示边界，内部直接共享 kernel，不必逐个调用公共 Op 拼装。

Text/MTP token 轴的合法范围由语义合同决定。私有 tile、网格和 route 阈值不能收紧公开范围；
实现可以拆分合法调用。Host envelope 只承诺 Device 值的范围，精确数学仍由设备输入决定。

## 代码组织

| 职责 | 内容 |
|---|---|
| Entry | 解释和检查公开输入，借用 workspace，调用实现 |
| Selector / dispatch | 按表示、几何、extent 和设备事实选择实现 |
| Shape policy | 某几何的生产实例与调优区间 |
| Launcher | 绑定模板、参数和 launch 资源 |
| Kernel | 数值计算、数据移动、同步与写入 |

这些是职责，不要求五个文件。简单 Op 可以在一个 `.cu` 中完整表达；复杂 family 才拆出 selector、
launch bindings 和共享模板。不要为一小段 helper 建空抽象或转发文件。

Linear 的 contraction 模板可以共享，projection、split、add、SwiGLU 等各自拥有调优记录与
分派，即使当前参数相同。通用 codec、设备数学与访存基础放入 `ops/common` 或 Core；
私有 shape 选择不进入共享文件。模型只包含语义头，不包含私有 kernel 或 selector。

Target 私有调优在对应 target 的目录和 source manifest 中选择；共享数学入口不接收 target key。
编译目标、ISA 与 RDC 约束由[构建系统](build-system.md)维护。不存在的实现由所属 Op 报错，
上层不复制 shape/support 清单。

命名表达数学和职责：`select_*` 选择、`*_dispatch` 分派、`launch_*` 提交。
Kernel 名称、模型标签、生命周期阶段不成为语义入口的组织依据。

## 头文件与资源

每个 Op 或紧密相关 overload 只维护一处必要说明：公式、表示输入、输出/状态效果、有效读写域、
alias 与资源要求。实际注册 shape 在实现中维护。头文件不记录调优历史、测试配方和 kernel 细节，
文档也不复制这些声明。

Observable 中间舍入属于合同；私有 accumulator、reduction tree、staging 和执行分解由实现决定。
随机 Op 显式接收 seed、逻辑位置及所需概率输入。Exact transform 或 ReplaySSM 的精确关系按其
专门合同说明，不能推广为所有浮点实现逐 bit 一致。

CUDA Op 接收 owning DeviceContext 传下来的 CudaExecutionView，CPU Op 借用 CpuExecutionView。
检查真实 operands 和 workspace 的位置；不在调用点从 tensor 与裸 stream 临时拼装执行环境。
CPU worker 不调用 CUDA；异步提交、跨 stream 复用与长期存储属于调用者。

Workspace 由调用者提供，Op 仅在 scope 中借用，不隐藏设备分配或持有返回后的 scratch。
有 scratch 的入口提供覆盖公开 extent 区间的容量查询，查询与执行共用选择和布局规则。
必须覆盖非单调 route 的高水位；不能只测区间上界。固定零 scratch 的入口省去 arena 和查询。
Host scope 退出不结束 GPU 使用，复用仍遵循 stream 与消费者寿命。

## 数值验证

验证公开合同和实际生产入口。数学 Op 使用独立 FP32/FP64 参考，按 public represented inputs
解码权重、执行完整公式；不复制生产路线的激活量化、内部 cast、reduction 或 workspace 精度。
融合参考计算完整公式，不拼接若干生产 Op 的舍入输出。模型输出正常、两条实现一致都不能替代数学参考。

Exact transform/codec 使用独立精确参考。ReplaySSM 等明确承诺与某次执行相同的合同，
使用同一初始状态、同一物理输入块和对应 trajectory 检查；较短输入的另一次投影不是该参考。
它的精确关系与原始 recurrence 的数学精度分别验收。

Criterion 由公开数值政策或表示决定，不由 target、私有精度、T 或某个 case 决定。
Linear-family 的 `A16Only`、`AllowA8`、`AllowA4` 分别使用 A16、A8、A4 要求，
即使某次执行采用更高精度也不切换标准。Criterion 明确 non-finite 规则；归约和矩阵类通常使用
normwise 误差并保留有限的 gross pointwise 上限，精确写域逐位检查。

有限 case 选择覆盖受影响入口、表示、生产路线边界和真实风险：state 历史、alias、stride、页边界、
无效行、零提交、untouched regions 及 Graph 更新。不要把性能扫描复制为常驻测试笛卡尔积。
Metadata 筛选先于大 fixture/设备初始化；相同表示的不可变 fixture 可以复用。
状态组合与资源寿命在模型/runtime 测试中检查，私有文件或 launcher 名称不是测试对象。
命令和 runner 组织见[tests](../../tests/README.md)。

## 调优与测量

先确定真实工作负载、计时边界、cache 条件和硬件参照，再比较候选。
SIMT/MMA 是执行机制，小 T/大 T 是负载区域，二者不绑定。低比特 MMA 所需的激活量化属于
完整路线，不能只报告 contraction 时间或忽略额外 launch/workspace。

候选空间已知时，一次构建有意义的候选×extent 集合，统一采样比较；不要反复改一项模板、
重编译再测少量点。临时强制路线和实验入口留在任务材料，选定后收敛到生产 selector，
通过公开入口确认最终数学与完整成本。

对 Linear，延迟重点为 T=1、4、8，并覆盖并发和 draft 产生的实际组合。
调优时查看 1..128 的逐点曲线，512/1024 为吞吐锚点；实际 Prefill 使用 2048/4096 等规模时
纳入对应点。其他 Op 按其真实 batch、history、state 或路由分布选择轴，不能只套用 Linear 的 T。

核心点单独判断，其他区间优先少量可复用策略。Dense 测量不要求 exact-T dispatch；
一次很小的抖动不值得增加模板实例。审查 route 交界、完整曲线和编译实例成本，
用实测误差判断变化，不用单点或加权平均隐藏明显低效区间。

Roofline 使用真实生产路线的必要有效 bytes、useful FLOPs 和对应硬件参照。
权重流量包含 codes/scales；MoE 按实际触及 expert 集合与 assignments；attention/SSM 按实际历史、
state 与必要 workspace。逻辑吞吐不是 DRAM 计数器或执行指令数，激活许可也不证明某种 MMA 已运行。
热点复用、冷热 cache、状态恢复和计时方式必须明确。

Op benchmark 证明 Op 性能；端到端收益需要相同模型负载的产品测量。
没有旧实现的新能力报告绝对性能和 roofline，不编造提升基线。增益与回退采用同一证据标准，
保留材料相关的异常和未达目标区间。

长期 benchmark 默认只调用公开 Op。GDN 现有 stage/crossover 测量是有意保留的内部诊断入口，
其结果须标明测量范围；不能代表完整 Op。其余 private candidate sweep 不进入长期工具。
命令、profile 和报告字段见[bench](../../bench/README.md)。
