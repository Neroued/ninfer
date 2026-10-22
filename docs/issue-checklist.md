# GitHub Issue 处理清单

审阅日期：2026-10-22。范围：按创建时间从新到旧的最近 50 条 Issue，
即 #415～#336，包含 42 条未关闭和 8 条已关闭的 Issue，不包含 PR。

依据 Issue 正文、全部评论和当前 `dev` 源码整理，各条目注明实际验证范围；
下文区分源码确认、待复现和旧实现已替换；报告者提出的根因或补丁仍需独立判断。

按下列顺序推进；相关问题可合并处理，但保留各自的完成状态。

- P0：可能使整个引擎失效的正确性问题，首先复现和定位。
- P1：加载启动故障、输出质量问题和明显的性能缺陷。
- P2：其他数值边界、性能、兼容性、统计和使用问题。
- P3：新增能力，需确定范围后再实施。

## P0：引擎失效

- [x] [#414 相同前缀并发触发 KV reservation 错误](https://github.com/Neroued/ninfer/issues/414)
  - 已定位旧调度漏洞：先为 A 预留并缓存可运行标记，再接入 B；B 的尾页复制 pin 住 A 的页，
    旧调度没有在执行 A 前重新检查阻塞，导致临时 COW 需求超过预留。
  - `71cdd615` 已通过执行前的 `unit_runnable()` 和 Session `context_blocks()` 覆盖此路径。
    与 `5e581e92` 的 Qwen4Exp prefill/replay 执行 frontier 修复不同。
  - 最小目录复现确认：A 页需求在复制前、复制期间、复制完成后为 0→1→0；
    忽略阻塞会触发相同报错，等待 B 完成后可沿用原预留执行。
  - 当前 27B/C4/MTP3/FP8/optimized proposal head 实测：8K/chunk128 与 131K/chunk4096
    两组配置共 20 请求成功，包含共享系统前缀与深前缀命中，服务保持可用。
    使用本地权重和自构造 prompt、512MiB Host cache；原报告未提供请求正文，未原样回放。
- [ ] [#399 grammar violation 导致整个引擎失效](https://github.com/Neroued/ninfer/issues/399)
  - 暂缓：原始触发根因待定位，保留以下排查结论；后续取得可回放请求或失败 token/state 后继续。
  - 原始非法 token 的根因尚未确定：报告没有可回放请求和失败 token/state；不能认定 `0a8530d4` 已修复。
    报告怀疑的多 repetition parent 上下界处理在 `81c8ce09` 就已存在。
  - 旧版与当前版的临时 CPU 对照均通过：1,981 个前缀状态、101,031 次候选检查，
    覆盖多分支重复、UTF-8 拆分、规范 JSON 转义、127/128/129 长度边界与 JSON Schema 转换。
    当前 grammar/JSON 合同测试通过；这些检查没有复现报告中的长工具请求。
  - 当前仍有确定缺口：输出预提交的 grammar 异常传播至 `fail_all_locked()`，使全部请求和引擎失效；
    普通服务错误日志不保留异常正文，只有请求 JSONL 保留原因。
  - 后续先区分模型 token 与强制 thinking control 的接受失败，补足 token、位置和 mask 诊断；
    若将明确的请求输出拒绝隔离到单行，需同步完善 Runtime 与两套模型 Session 的提交、回收合同。
    现有 `CommitDecision.failed` 只表示 GPU 已报告的约束死路；不能直接复用或捕获所有 `logic_error` 后继续。

## P1：启动、输出与明显性能缺陷

- [ ] [#391 第二张 GPU 上 NVFP4 长 prefill 崩溃](https://github.com/Neroued/ninfer/issues/391)
  - 暂缓：当前没有报告中的 Proxmox/VFIO 复现环境。
  - 根因待定位：Proxmox/VFIO 下两张 5090 分别运行独立容器，其中一张在长输入或图片请求下失败；
    报告称 NVFP4 触发 Xid 43、groupwise-int 正常，未提供代码版本、失败请求、prefill chunk 或首个失败 kernel。
  - `cudaStreamSynchronize` 是异步错误暴露点，当前 `CUDA_CHECK` 仍会 abort；该行不是失败 kernel 的位置。
    报告同期主线和当前 prefill 都走 eager，评论中的“大 prefill Graph”归因没有源码依据。
  - 首先在报告环境用最短失败请求、eager 与 `CUDA_LAUNCH_BLOCKING=1` 取得首个失败 launch；
    若指向 NVFP4，再缩小为对应 shape/T 的 A4 量化、MMA/TMA 算子复现。
    输入总长度不等于 linear 的 M；现有证据不足以归因于某个分派、驱动或物理设备，也不能标记已修。
- [x] [#392 JSON object 首字段异常](https://github.com/Neroued/ninfer/issues/392)
  - 已在当前 5090 / 27B NVFP4 / FP8 KV / MTP3 / thinking off / greedy 上复现并隔离。
    `src/text/json_schema.cpp::build_json_grammar()` 使用 `any_whitespace=false` 和紧凑分隔符，
    排除了结构处的空格、换行及携带它们的合并 token；CPU 检查确认两种 JSON 入口都存在此限制。
  - 三个短 prompt 各对照四种输出方式，共 12 请求：原生 `json_object` 与等价紧凑 GBNF 均出现
    空 key 或错误值结构；仅打开 whitespace 的 GBNF 三例均恢复预期字段和值。
    GBNF 对照通过正常服务接口完成，没有修改生产代码；无约束对照也生成了预期字段和值。
  - 已修复通用 JSON grammar：允许标准结构空白与正文首尾空白，不设置额外长度上限；
    对象/数组 const、enum 同样支持格式空白，保留字符串和数值字面量，Qwen XML 工具格式保持自身合同。
  - 完整项目编译及 7 组 grammar、JSON 独立参考、工具组合和协议测试通过。
    原生接口复测三个复现 prompt，以及 JSON Schema 流式、固定容器值、thinking，共 6 请求全部正确完成。
    `json_object` 本身仍只约束 JSON 根对象，不保证 prompt 指定的字段和值。
- [x] [#404 ZFS Direct I/O 对齐与加载规划不一致](https://github.com/Neroued/ninfer/issues/404)
  - 根因：materializer 固定按 4KiB 规划，而文件层按 `statx` 的实际要求校验。
    已改为先取得各文件 FileView，再按实际要求规划、合并与分块，保留合法的 Host 原位读取。
    编码布局和 4KiB 容器 framing 不变，不降低文件层报告的对齐要求。
  - 修复前已通过注入 128KiB offset/length 要求复现；修复后的回归覆盖 8KiB memory、128KiB offset/length、
    跨文件窗口合并及 EOF 尾部。未在真实 ZFS 上实测。
- [x] [#372 小 pinned staging 缺少文件读取对齐保证](https://github.com/Neroued/ninfer/issues/372)
  - Slot 按文件的 memory alignment 提供对齐视图；pinned 分配额外对齐余量并保留原始 owner，
    普通 Host staging 显式对齐分配，staging 统计包含余量，不修改全局 pinned buffer 合同。
  - 已验证故意偏移的 pinned 地址、CPU/GPU 分发，以及 event 创建/记录失败时的上传完成和资源释放。
    完整项目编译及 artifact、I/O、PLE 相关 10 项测试通过。
- [x] [#406 Docker 构建与运行依赖遗漏](https://github.com/Neroued/ninfer/issues/406)
  - 已在 `1e902680` 补齐 liburing/JPEG 构建与运行依赖、构建阶段 CUDA Driver stub 链接、
    运行镜像项目共享库打包，并排除 `.agent-tmp` 构建上下文。
  - 按用户要求，未运行构建或运行验证。
- [ ] [#378 Host 总空闲字节不代表实际可放置](https://github.com/Neroued/ninfer/issues/378)
  - 当前仍存在按总空闲字节判断、实际申请连续 extent 的衔接风险。
  - 保留迁移必须取得可落实的目标空间，避免迁移失败后释放原本要保护的 conversation。
    先解决决策与分配合同，再考虑 #379 的 allocator 组织。
- [ ] [#394 缓存满后多轮对话只命中系统前缀](https://github.com/Neroued/ninfer/issues/394)
  - 当前策略待复现：检查复用证据、可选 checkpoint 准入和回收收益是否长期阻止更深 turn closure 入驻。
  - 评论已指出简单的“第二次访问才保存”方案存在统计时序问题，不直接采用该补丁。
- [ ] [#356 NVFP4 tiled prefill 缺少 split-KV](https://github.com/Neroued/ninfer/issues/356)
  - 源码确认：当前 tiled launch 按 query tile × head 启动，每个 CTA 遍历完整 KV。
  - 在当前模板中接入 KV 分区与合并，改善长上下文利用率，保留既定 QK 精度；
    与 #352 的额外 QK 量化提案分开。

## 历史严重故障：先取得当前主线证据

以下报告影响严重，但来源为旧 Windows 移植或已被替换的实现，暂不作为当前主线已确认故障。
复现后按实际影响转入 P0/P1。

- [ ] [#357 Host restore 悬空事件视图](https://github.com/Neroued/ninfer/issues/357)
  - 指向旧 `HybridPrefixCache/RestoreBatch` 生命周期；当前对应路径已替换。
    需当前受支持环境的故障证据，不迁入旧容器补丁。
- [ ] [#339 并发资源减法下溢](https://github.com/Neroued/ninfer/issues/339)
  - 旧资源账本与实现已替换。保留为 #414 资源预留检查的历史场景，尚无当前主线复现。
- [ ] [#393 休眠恢复后 pinned RAM 重复](https://github.com/Neroued/ninfer/issues/393)
  - 来自 Windows 移植，当前主线没有报告描述的自动 CUDA context 重建流程。
    需先确认对应资源生命周期与主线适用性。

## P2：其他修正与优化

- [ ] [#349 proposal 采样的 CDF 边界](https://github.com/Neroued/ninfer/issues/349)
  - Issue 已关闭，原消费端修复建议不成立；但当前 selector 在 CDF 未命中时仍退回最后候选，
    存在极低概率选中零概率项的边界风险。修正采样生产端的累计总量和合法回退，
    不把 `q<=0` 直接解释为接受。
- [x] [#405 `generation_service.h` 缺少 `<span>`](https://github.com/Neroued/ninfer/issues/405)
  - 已在 `1e902680` 直接包含 `<span>` 和 `<string_view>`，消除间接头文件依赖。
    按用户要求，未运行编译验证。
- [ ] [#371 取消或编辑后续请求丢失上一 continuation](https://github.com/Neroued/ninfer/issues/371)
  - 当前仍有消费原 checkpoint、取消时清理活动序列的路径。
    检查上一完整 turn closure 的保留；中间 checkpoint 能减少重算，但不等于此问题已解决。
- [ ] [#379 Host arena 碎片](https://github.com/Neroued/ninfer/issues/379)
  - 当前为连续 extent 分配，先完成 #378，再判断是否需要调整分配器。
    不预设统一 block pool 为答案，保留 StateImage 连续布局合同。
- [ ] [#364 NVFP4 down-projection 小 T 利用率](https://github.com/Neroued/ninfer/issues/364)
  - 当前 `[5120,17408]` A4 小 T 仍使用 M32N64。对比 M32N32 候选，
    根据当前 RTX 5090 算子与实际调用规模决定路由，不直接推广到 Spark。
- [ ] [#370 CPU converter job 并行](https://github.com/Neroued/ninfer/issues/370)
  - 当前权重 job 串行生产，已有读取预取和算子内部并行。
    评估 job 并发收益，同时考虑临时张量峰值与 CPU 线程竞争。
- [ ] [#381 Direct I/O 读模式与短读处理](https://github.com/Neroued/ninfer/issues/381)
  - Issue 已关闭，但当前物化仍固定 Direct I/O，`read_direct` 仍为单次 `pread`。
    随 #404 检查短读合同；云存储 buffered 读取作为独立需求，不直接改变本地默认策略。
- [ ] [#373 无 thinking budget 时计数为零](https://github.com/Neroued/ninfer/issues/373)
  - 源码确认：semantic thinking 跟踪仍依赖预算是否存在。
    将模型思考计数与预算控制分开，保留模型生成和引擎注入 token 的区别。
- [ ] [#341 HTTP 端口关闭后不能立即复用](https://github.com/Neroued/ninfer/issues/341)
  - 源码确认：Linux 上沿用 httplib 默认 `SO_REUSEPORT`，未同时设置 `SO_REUSEADDR`。
    在 HTTP transport 所属位置处理。
- [ ] [#401 上下文超限错误文案](https://github.com/Neroued/ninfer/issues/401)
  - 保留 HTTP 400 和 `context_length_exceeded`，改进文案以包含最大长度与实际长度，
    方便客户端识别和压缩重试，不改成 413。
- [ ] [#395 长 MCP 工具名兼容](https://github.com/Neroued/ninfer/issues/395)
  - 决定是否扩展 OpenAI 路径当前的 64 字符限制。若扩展，同步输入验证、生成约束和历史处理。
    OpenAI 与 Anthropic 上限不同本身不是 bug。
- [ ] [#348 Codex Responses 字段兼容](https://github.com/Neroued/ninfer/issues/348)
  - 分开评估 `reasoning.summary`、`include` 等字段与 `web_search` 工具能力。
    后者需要明确执行责任，不以接受后静默忽略代替支持。
- [ ] [#376 KV headroom 可配置](https://github.com/Neroued/ninfer/issues/376)
  - 区分自动容量余量和启动总预算安全余量，明确显式 KV 容量的计费行为后再设计配置。
    当前 1GiB 默认值保持既定含义。

## P3：后置功能候选

本组尚未选定实施范围；排列不表示已承诺开发。

- [ ] [#415 缓存检查与主动驱逐 API](https://github.com/Neroued/ninfer/issues/415)：
  检查能力与驱逐权限分别设计；释放逻辑条目不会自动归还启动时固定分配的 pinned backing。
- [ ] [#400 报告模板工具调用能力](https://github.com/Neroued/ninfer/issues/400)：
  先定义能力含义，单次模板探测不足以证明完整工具往返支持。
- [ ] [#361 拼接已编码 DFlash2 组件](https://github.com/Neroued/ninfer/issues/361)：
  artifact 工具能力；编码保值、config 和 tokenizer 相同不证明 draft 训练兼容性。
- [ ] [#413 EAGLE4](https://github.com/Neroued/ninfer/issues/413)：新 speculative backend，需独立设计与收益评估。
- [ ] [#375 高精度近期 KV tail](https://github.com/Neroued/ninfer/issues/375)：
  新存储与计算策略，包含额外显存、提交和回滚成本；已有测量只代表实际生效的格式。
- [ ] [#350 磁盘 L3 KV cache](https://github.com/Neroued/ninfer/issues/350)：
  新资源层，涉及持久身份、State/KV 一致性、I/O 生命周期和保留策略。
- [ ] [#362 主动检索 KV 窗口](https://github.com/Neroued/ninfer/issues/362)：
  改变模型可见历史，需要质量评估，不归入前缀缓存修复。
- [ ] [#383 非因果决策模型](https://github.com/Neroued/ninfer/issues/383)：
  新 attention 语义、readout、执行用途和服务协议。
- [ ] [#367 Nix Flake](https://github.com/Neroued/ninfer/issues/367)：构建分发扩展。
- [ ] [#336 AMD/ZLUDA](https://github.com/Neroued/ninfer/issues/336)：新硬件支持范围，需单独决策。

## 已覆盖或无需新增开发任务

此处为本轮技术处置结论，不代表已经修改 GitHub 状态。

| Issue | 结论 |
|---|---|
| [#402 回收评分三重扫描](https://github.com/Neroued/ninfer/issues/402) | 旧 `recovery_loss` 已被公共 `RecoveryIndex/RecoveryEvaluation` 替代，不迁入旧 PR #403。 |
| [#390 Spark 支持](https://github.com/Neroued/ninfer/issues/390) | 当前原生构建、加载、分派、单机与双机执行已覆盖。 |
| [#385 DFlash2 Graph 预算过大](https://github.com/Neroued/ninfer/issues/385) | 旧 allowance 估算已被启动实测校准替换，尚无当前实现复现。 |
| [#380 并发重发未等待 checkpoint](https://github.com/Neroued/ninfer/issues/380) | 已关闭；报告者已确认后续主线修复。 |
| [#369 自动共享系统前缀](https://github.com/Neroued/ninfer/issues/369) | 当前 Frontend 结构边界发现与共享前缀保留已覆盖。 |
| [#360 StateImage 字节计数](https://github.com/Neroued/ninfer/issues/360) | 当前按 `TransferWork.payload_bytes` 计数，评论已有真实字节复测结果。 |
| [#358 metrics endpoint](https://github.com/Neroued/ninfer/issues/358) | 已实现，Issue 已关闭。 |
| [#412 RTX6000 Pro 咨询](https://github.com/Neroued/ninfer/issues/412) | 已关闭，无具体待修复故障。 |
| [#386 oldest resident 无法取得资源](https://github.com/Neroued/ninfer/issues/386) | 已关闭，维护者明确属于 fork 问题。 |
| [#377 从未闭合思考段恢复工具调用](https://github.com/Neroued/ninfer/issues/377) | 维护者已明确不采纳；思考段不作为工具执行来源。 |
| [#374 tile 占用影响 FMA 舍入](https://github.com/Neroued/ninfer/issues/374) | 不同实现末位差异不自动构成数学合同违约；当前 epilogue 也已重构，不新增跨宽度逐位一致要求。 |
| [#366 缓存重写公告](https://github.com/Neroued/ninfer/issues/366) | 信息汇总，无独立开发任务；具体问题归入对应 Issue。 |
| [#354 NVFP4→FP8 staging 开销](https://github.com/Neroued/ninfer/issues/354) | 已关闭，分析针对未纳入的候选路径，保留参考。 |
| [#352 NVFP4 使用 8bit QK](https://github.com/Neroued/ninfer/issues/352) | 已关闭；引入额外 Q/K 舍入，若重启提案需独立决定计算策略，不替换当前精度语义。 |
