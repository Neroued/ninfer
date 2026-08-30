# Architecture、Config 与 Weight Realization 解耦讨论笔记

> 状态：进行中的临时讨论笔记。本文不加入文档索引，不是实施方案，也不替代现有
> architecture、artifact 或 target 权威。讨论收敛并完成相应实现后，应将稳定契约回写到
> 相关 active maintainer 文档并删除本文。

## 1. 讨论目标

当前 NInfer 以精确的 `(model_id, weights_id)` 选择 target-private `WeightsProfile`。
一个 profile 同时决定 checkpoint identity、完整 tensor inventory、逐层格式、物理融合边界、
binder、部分 workspace 上界和 execution leaf 路径。这种设计允许对少量固定 artifact 做深度
优化，但每增加一种逐层混合精度或 groupwise 组合，都可能需要新增一个完整 profile，模型数学
定义与物理权重实现因而被绑定在一起。

本次重构讨论的目标是：

```text
Architecture Definition
    + explicit NInfer-owned Architecture Config
    + hierarchically resolved Weight Realizations
    -> one load-time validated, execution-stable Program
```

Architecture 继续直接定义模型数学、状态和调度；artifact 明确给出 config 和已经转换好的物理
权重；引擎在每个语义执行点验证并绑定一个已有 realization。目标不是建立任意模型图、动态图
优化器或未知权重插件系统。

## 2. 已确定的概念

### 2.1 Architecture Definition

Architecture 表示代码级数学定义和算法结构，命名可以沿用 vLLM/Transformers 的 architecture
名称，例如：

- `Qwen3_5ForCausalLM`；
- `Qwen3_5MoeForCausalLM`。

已经确定：

1. Qwen3.5、Qwen3.6 和 Qwen3.8 是 checkpoint/release 系列，不是不同 architecture。
2. Qwen3.5/3.6/3.8 27B 属于同一个 dense architecture；相同数学 config 不应因 release 名称
   进入不同执行实现。
3. 9B、27B 等尺寸属于同一 architecture 下的不同 config。
4. Qwen3.5/3.6 35B-A3B、122B-A10B 等属于同一个 MoE architecture 下的不同 config。
5. Dense 与 MoE 的数学定义不同，因此是不同 architecture。
6. checkpoint/release 名称不再作为执行代码或 target package 的选择轴；是否保留非执行性的显示
   或 provenance 字段不是当前核心决策。

### 2.2 Architecture Config

Architecture Definition 给出公式和结构规则；Architecture Config 给出一个具体模型实例的拓扑、
尺寸和数学参数。

已经确定：

1. `.ninfer` 必须新增显式 architecture identifier 和完整结构化 config。
2. architecture identifier 可以使用 vLLM/Transformers 已采用的命名。
3. config 契约由 NInfer 自己拥有和维护。转换器可以从 Hugging Face `config.json` 取得初始数值，
   但 NInfer 定义字段集合、类型、默认值、验证、规范化和后续演进语义。
4. 不允许 loader 根据 tensor shape、对象数量、对象名称或代表性 descriptor 猜测 config。
5. 计算图定义和权重 binding 架构应当能够接受任意明确声明且在数学上合法的 config。
6. 实际能够执行的 config 可以继续是有限集合；限制应来自所需 Op 的 shape/state 支持以及所有
   weight realization 的可用性，而不是来自 Qwen3.6、Qwen3.8、9B、27B 等 checkpoint-specific
   target 分支。
7. 每个 Architecture Config 必须覆盖该 architecture 自身的全部数学事实。Primary model config
   覆盖适用的 Text、Vision 和 MoE 参数；作为可选 speculative module 附着的 MTP、DFlash 等拥有
   各自独立的 architecture/config，不塞入 primary model config。

具体 schema 字段将在实施设计中从模型数学和已有 config 校验代码推导；这里不提前定义 JSON
spelling、字段分组或 schema version 机制。当前 v2 容器无法承载该 config，因此容器版本必须
提升；具体版本号和 framing 属于实施细节。

### 2.3 Weight Realization

Weight Realization 表示一个语义执行点所采用的物理权重形式，包括但不限于：

- numeric format 与 storage layout；
- 一个逻辑投影由一个 fused parent 还是多个 split parents 提供；
- fused parent 的逻辑 row 顺序与零拷贝 views；
- scale、divisor 和其他格式私有 persistent values；
- 与该物理表示兼容的已有 Op/kernel 路径。

已经确定：

1. Semantic site 与 physical form 是不同概念：前者属于 architecture 数学结构，后者表示 artifact
   为该语义选择的物理权重与实现形态。
2. 模型语义采用分层抽象。一个 GDN layer 可以包含 projection、norm、attention/GDN core、FFN
   等子结构；层、子结构和更细的闭合操作形成语义层级，而不是一个预先固定的平面 site 列表。
3. Realization 的选择尺度不能固定为“整层”或“每个 tensor”。具体边界取决于数学图、已有实现、
   权重的物理 grouping 和可保留的 fusion。一个受支持 realization 可以覆盖 architecture 明确允许
   的闭合语义区域；没有较大 fused realization 时，同一区域可以继续由更细粒度 realization 组成。
4. 分层语义不能限制未来 fusion，但 artifact 也不能任意跨越 architecture 未注册的语义边界。
   每个可选择区域必须具有稳定的输入、输出、状态与权重 ABI，并由 architecture 拥有其允许的
   physical forms。
5. 不同层和不同语义区域可以组合不同的已有 realization；新的混合分配不应仅因为组合不同就要求
   新增一份完整模型 Program、target package 或 `WeightsProfile`。
6. artifact 只能选择引擎已经实现的 realization。若需要新的 fusion、format、layout 或执行 ABI，
   artifact 与引擎必须同步修改；artifact metadata 本身不能创造新执行能力。
7. 因而目标是“闭合能力集合上的分层组合”，不是让引擎解释任意未知物理格式。

语义层级、可选择区域和 payload ABI 的最终清单尚未设计。它们应根据数学状态/舍入边界、共享
activation、已有 fused Op contract、workspace 和性能需求推导，而不是机械地按 layer class 或
每个源 checkpoint tensor 拆分。

### 2.4 Architecture-owned Physical Forms

已经决定采用 architecture 与任意 artifact mapping 之间的中间边界：

```text
Architecture(config)
    -> 定义分层语义和可绑定的闭合区域
Artifact
    -> 为选定区域声明一个 architecture 已注册的 physical form 并引用 physical objects
Object descriptors
    -> 提供 shape、numeric format 和 storage layout
Engine
    -> 验证完整组合并解析为已有 realization
```

具体 metadata spelling 尚未设计，但边界已经确定：

1. Architecture 拥有 semantic ABI、允许的 physical-form ABI、parent 数量、logical row order、
   辅助 scalar 和合法零拷贝 views。
2. Artifact 显式选择受支持 form 并绑定已经生成好的 physical objects；不依赖完整 `weights_id`
   选择 inventory，也不靠对象存在性猜测 config。
3. Tensor descriptor 继续表达实际 format、layout 和 shape。Artifact 不直接指定 kernel。
4. Engine 根据 semantic region、form、formats/layouts、shape、hardware 和所有可达执行模式解析
   具体 realization。
5. Artifact 不携带任意 slice/reshape/concatenate/transform graph。新增一个未注册 physical form
   需要同步扩展 architecture contract、converter、binder 和执行实现。

### 2.5 删除 `weights_id`

`weights_id` 不再需要，也不再参与 `.ninfer` 执行身份或 registry 选择。执行所需事实由
architecture、NInfer-owned config、physical-form bindings 和 object descriptors 完整给出。

若转换流程仍需要 `recipe_id`、人类可读 profile label 或 provenance，它们属于 conversion report
或其他非执行 metadata，不能恢复为选择 binder、workspace 或 Program 的权威。

## 3. Artifact 与 Loader 的已确定契约

### 3.1 自由度发生在不同 artifact 之间

一个 artifact 固定一种完整物理权重表示。相同 architecture/config 可以有多个 artifact，每个
artifact 在各语义执行点选择不同的受支持 realization；同一个 artifact 加载后不重新选择另一种
权重格式。

例如，同一个 GDN input projection 可以分别由不同 artifact 存储为：

```text
query_key(Q4) + value_z(Q5)
query_key_value_z(NVFP4)
query_key_value_z(FP8)
```

三者表达同一数学执行点，但物理 parent、format 和 kernel 路径不同。

### 3.2 所有权重转换发生在 artifact 生成阶段

Converter/artifact generator 负责量化、fusion/split、packing、layout 转换、scale/divisor 生成和
物理 parent row order。生成后的 artifact 已经是引擎可消费的物理表示。

### 3.3 Loader 不修改权重

Loader 只负责：

- 读取并验证 architecture、config、object descriptors 和 realization binding；
- materialize/upload 已有 payload；
- 建立 slice、row view 或其他不生成新权重字节的零拷贝 view；
- 将每个已验证语义执行点绑定到已有实现。

Loader 不执行：

- fuse、拆分后复制或合并权重；
- transpose、swizzle、packing 或 repack；
- 量化、反量化或重新量化；
- 为另一个 kernel 生成第二套权重字节。

### 3.4 Realization 在加载期解析

所有 realization 必须在加载/规划期完成解析，而不是在首次请求或某个罕见执行 phase 中延迟发现：

```text
read architecture/config
    -> enumerate semantic hierarchy
    -> bind physical forms and objects
    -> resolve every reachable realization
    -> validate cross-region constraints
    -> plan workspace and persistent resources
    -> materialize weights
    -> construct Program and CUDA Graphs
```

Program 可以持有不可变 enum/variant 或其他直接 dispatch 数据，并允许执行时做简单 dtype/form
分支；它不重复解析 strings、object descriptors 或 capability rules。Workspace 和 CUDA Graph
规划必须依据实际解析结果，而不是一个完整 artifact `WeightsProfile` 的保守替代。

### 3.5 Trusted Artifact 边界

NInfer 继续信任本项目生成、验证和注册工作流产生的 artifact。开放更多 config 和 realization
组合是为了降低新增可信 artifact 的工程成本，不是为了接受不可信第三方输入。

因此：

1. Loader 仍验证容器 framing、显式 config、object descriptors、physical-form ABI、shape/format/
   layout compatibility、完整 binding 和实现可达性，因为这些是正确执行与清晰诊断所必需的。
2. Loader 对不支持或不匹配的 artifact 直接失败，不猜测、不修复、不转换，也不提供 fallback。
3. 不为敌意 metadata、恶意 payload、沙箱隔离、签名、通用自描述输入或任意损坏恢复扩大设计。
4. Converter、资格验证和发布工作流负责在 artifact 到达 loader 前证明其内容满足项目合同；运行时
   校验不替代该可信生产链，也不演变成面对任意输入的防御框架。

## 4. 执行模型的已确定边界

“避免动态图”在本讨论中表示：

- 不建立大型通用 compute-graph IR；
- 不执行类似 llama.cpp 的 graph optimization passes；
- 不在运行期解释任意字符串节点或未知 graph topology。

允许：

- 直接由 architecture C++ 实现按照显式 config 执行固定数学调度；
- 根据已绑定 weight dtype、payload variant 或 realization 做直接分支；
- 在加载完成后形成稳定执行路径；
- 对稳定路径执行 CUDA Graph capture/replay。

模型数学图只由 architecture/config 决定。Weight realization 可以改变 projection 的拆分/融合、
kernel 数量、workspace 和最终物理 kernel DAG，但不得改变可观察数学与状态语义。

## 5. 模型组成与内嵌 Speculative Modules

### 5.1 统一权重机制，区分组合关系

Text、Vision、MTP、DFlash、DSpark、DFlash2 等都使用同一套权重抽象：

```text
semantic execution point
    -> physical Weight Realization
    -> supported implementation
```

它们不发展彼此独立的 artifact format、loader 框架、weights profile 或 binding 原理。但统一底层
机制不代表它们具有相同的组合关系和生命周期：Text/Vision 可以属于 primary model 本身；MTP、
DFlash 等候选解码器被建模为附着于 primary model 的可选、内嵌 speculative module。

### 5.2 Artifact 组成

一个 artifact 包含一个必需的 primary model，以及零个或多个 speculative module：

```text
Artifact
├── Primary model
│   ├── architecture/config
│   └── semantic weights and physical forms
└── Speculative modules [0..N]
    ├── official-mtp
    ├── community-dflash
    ├── community-dflash2
    └── ...
```

Speculative module 不是 primary model 的 weight realization，也不是运行时外挂的第二个 artifact。
它是当前 artifact 内一个自包含的 auxiliary model instance。这样同一个 artifact 可以携带官方
MTP head 和一个或多个社区方案，同时只保存一份 Text 权重，避免外挂权重和重复下载主模型。

每个 speculative module 必须显式给出：

1. module architecture，表示其数学拓扑、proposal 算法和状态语义；
2. NInfer-owned module config，表示其层数、维度、feature layers、draft window、私有 KV/state 等
   具体数学事实；
3. attachment contract，表示它消费 primary model 的哪些语义 feature taps、引用哪些共享权重、
   支持哪些执行模式，以及如何参与 propose/verify/accept/commit/rollback；
4. module 自有 semantic bindings、physical forms 和 private objects。

Module architecture 与 weight realization 继续正交：数学公式、拓扑、状态或 proposal protocol
不同的 MTP、DFlash、DSpark、DFlash2 是不同 module architecture；同一算法仅有 BF16、FP8、
NVFP4、groupwise 或融合布局差异时，是同一 module architecture 的不同 physical realization；
同一算法重新训练得到的参数集，是另一个 module instance，而不是新的 backend architecture。

### 5.3 Attachment 与共享对象

Primary architecture 显式定义有限、稳定的 attachment sites 和语义 ABI。Module 只能引用这些
注册过的 feature taps、target verification 接口和共享权重，不能根据 tensor 名称或 shape 猜测
如何挂接，也不能携带任意计算图改变 primary model 的数学语义。

Module 可以通过显式 artifact 内引用复用 primary model 的 embedding、LM head、feature outputs
或其他允许共享的对象；多个 module 也可以引用同一个 proposal head。对象 payload 只存储和
上传一次，不为每个 module 复制。Loader materialize 的是：

```text
primary model + selected speculative module 的对象依赖闭包
```

跨组件引用只建立零拷贝 binding/view，不允许在加载期生成、融合或重排共享权重。

### 5.4 Module Instance 选择与驻留

每个 speculative module 具有 artifact-local module name，用于区分同一 artifact 内的多个实例并
供 Engine 启动参数选择。该 name 只是容器内选择句柄，不是 `weights_id`，不参与 kernel、binder
或 realization 的权威判定，也不要求跨 artifact 稳定。

已经确定：

1. 一个 artifact 可以携带零个、一个或多个 speculative module，也可以携带同一 module
   architecture 的多个训练实例。
2. 每个 Engine 实例在启动时显式选择 `none` 或至多一个 module。
3. 只对被选 module 解析完整 execution capability、规划 workspace/CUDA Graph、上传其依赖对象并
   构造私有状态；未选择 module 不占 GPU 常驻空间。
4. 选择在 Engine 实例生命周期内不可变，不支持逐请求选择、同时执行多个 backend 或运行中切换。
5. Artifact 中存在多个 module 解决的是发布、下载、共享主模型和选择问题，不扩大当前
   startup-fixed Program 与资源模型。
6. 被选 module 不存在、attachment 不匹配或引擎没有对应 realization 时，加载直接失败，不进行
   fallback 或近似替换。

Runtime 可以在 target verification、accept/commit 和 token publication 等事务边界建立统一
speculative protocol，但各 module architecture 可以分别拥有自己的 propose schedule、feature
extraction、KV/state、workspace、CUDA Graph frontier 和直接 C++ 实现；不为统一接口建立通用动态
计算图。

### 5.5 与 Primary Weights 的精确配对

社区 speculative weights 往往依赖训练时使用的精确 primary checkpoint，而不只是相同的
architecture/config。当前可信、自包含 artifact 模型下：

1. module 的 attachment 显式指向同一 artifact 内的 primary model；
2. artifact generator 负责确认 module 与所携带 primary weights 的精确来源匹配，并完成转换期
   资格验证；
3. runtime 信任这一原子组合，不为每次加载重新计算全模型内容哈希；
4. checkpoint revision、source digest 等可以记录为生成/验证 provenance，但不成为 execution
   identity 或新的 `weights_id`。

当前不支持把独立下载的 speculative weight package 在运行时挂接到已有 artifact。只有未来明确
引入跨 artifact 组合时，才需要设计跨文件的 primary-weight content fingerprint 和兼容协议。

## 6. 预期可支持的代表性行为

重构后的设计应能够表达以下结果：

1. Qwen3.6-27B 与 Qwen3.8-27B 在相同 config 下进入同一个 architecture 实现，不再因 release
   名称分叉。
2. 一个逐层混合 NVFP4、FP8 和 BF16 的新 artifact，只要每个执行点都使用已有 realization，
   就不需要注册一个新的完整 target/profile。
3. 一个由已有 groupwise formats 和 fused Ops 构成的新组合，只需要 artifact 选择相应 realization，
   不复制模型计算图和 Program。
4. 9B 或其他 config 在所有必需 Op/realization 支持后可由同一 architecture 实现执行，而不新增
   checkpoint-named architecture。
5. 不支持的 config、shape、fusion、format、layout 或全局组合在加载/规划期间明确拒绝，不能在
   首次请求的 GPU 执行中才偶然失败。
6. 已有 artifact-native fusion、专用 kernel 和深度优化路径仍能作为 realization 保留；解耦不能
   强迫所有权重降级为通用 linear graph。
7. 一个 artifact 可以内嵌 MTP、DFlash、DSpark 等多个候选 module，共享同一份 primary weights；
   Engine 启动时选择其中一个而不为每个候选复制 Text payload 或占用 GPU 常驻空间。

## 7. 当前实现与目标之间的已知差距

当前 `.ninfer` v2：

- JSON root 严格只有 `identity` 和 `objects`；
- `identity` 严格只有 checkpoint-native `model_id` 和完整 storage-contract `weights_id`；
- 不保存 architecture identifier 或模型 architecture config；
- tensor descriptor 保存 name、shape、format、layout、offset 和 bytes，但不保存显式逻辑
  role/fusion/view mapping。

当前 converter 会读取并严格验证 Hugging Face config，但只把 `config_summary` 写入外置
`.conversion.json`。运行时使用的 layer count、dimensions、topology 和模型常量仍由 target
`config.h` 等编译期代码提供。

当前 27B `WeightsProfile` 同时决定：

- `(model_id, weights_id)` 接受范围；
- endpoint 和逐层 numeric formats；
- split/fused object inventory；
- binder 分支；
- 部分 workspace capacity policy。

现有实现中可继续利用的基础包括实际 `Weight.qtype`、split/fused payload variants、按 dtype/policy
分派的部分 Ops，以及 architecture family 共享 schedule。

因此本次方向至少要求扩展或替换当前容器 metadata 契约，并重新划分 registry、config、binder、
model view、workspace planning 和 execution leaf 对完整 `WeightsProfile` 的依赖。

## 8. 留待后续实施设计

以下项目不再是当前产品方向决策，但需要在正式实施设计中确定：

1. Qwen3.5 dense/MoE 的完整语义层级，以及哪些闭合区域允许 parent-level fused realization、
   哪些区域只由子 realization 组成。
2. 每种 physical form 的 typed binding 字段、parent/slice ABI、辅助 scalars、alias 和跨区域约束。
3. `.ninfer` 中 architecture/config/form bindings 的具体 JSON schema、framing 版本与 C++/Python
   owning types。
4. Loader capability resolver、不可变 dispatch representation、workspace aggregation 和错误诊断。
5. 当前五个已发布 artifact 向新容器契约的统一转换，以及旧 `model_id/weights_id` registry、
   `WeightsProfile`、binder 分支和 profile-based workspace 路径的删除顺序。
6. Context-cost、benchmark 和诊断工具所需的 canonical config/realization signature；该 signature
   从解析结果派生，不成为新的手写 `weights_id`。
7. Primary/module inventory、artifact-local module name、attachment contract、共享对象引用和
   selected-module dependency closure 的具体容器 schema 与 typed C++ 表达。
8. 当前硬编码 `SpeculativeBackend`、`optional<MtpLayer>`、`optional<DFlashPayload>` 和对应
   workspace/graph 分支向 module inventory + load-time resolved immutable backend 的迁移方式。

精确层级和 form 设计应以现有 Qwen3.6/Qwen3.8 dense 与 Qwen3.6 MoE 路径为代表性输入，但不能
再次把这些 checkpoint artifact 的当前差异提升为 architecture identity。

## 9. 明确不在当前方向中的事项

- 根据文件名、tensor shapes 或 inventory 猜测 architecture/config；
- 加载期 weight fusion、repacking 或 quantization；
- 同一个 artifact 在加载后切换成不同物理权重格式；
- 运行时外挂独立 speculative weights，或把未经 artifact generator 配对验证的 module 附着到
  primary model；
- 在同一个 Engine 实例内逐请求选择、同时执行或运行中切换多个 speculative module；
- 通用动态模型图、graph optimizer 或 string-driven execution；
- 仅因 Qwen3.5/3.6/3.8 release 名称不同而复制 architecture runtime；
- artifact 声明一个引擎未知 realization 并期望自动获得执行能力；
- 为不可信或恶意 artifact 建立沙箱、自动修复、格式猜测或兼容 fallback。
