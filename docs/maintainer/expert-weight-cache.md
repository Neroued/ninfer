# Expert 缓存与混合执行

RTX 5090 的 MoE offload 在完整 Host expert 来源与有限 GPU 副本之间分工。
Spark 使用 GPU 模型计算。公共 MoE 机制位于 [models/common/moe](../../src/models/common/moe/)，
模型仍拥有层连接、Main/MTP 放置和逻辑 expert 身份。

## 权重与计算边界

ModelShard 持有不可变权重来源和注册寿命；RankProgram 持有 GPU cache、预取、CPU bridge 与
实际调度。缓存不属于请求 State，不随 prefix restore、暂停或 speculative 拒绝回滚。
Shared、router、MTP 等固定部分留在 GPU，Main routed experts 按资源计划放置。

CPU/GPU 消费同一次全局路由和原始系数，子集分工不重新归一化。Logical expert ID 与物理 slots
独立，未选中的地址不被读取。完整 GPU 路线继续使用已优化的融合 Op；混合路线分别计算分项，
按共同输出合同汇合，不重复加入 shared 或 residual。

Bank/codec 由 [Weight views](../../src/core/weight_bank.h)和
[MoE Op](../../include/ninfer/ops/moe.h)维护。缓存只发布编码副本，不做数学或转换。
CPU 直接读 Host bank；注册视图供异步 CUDA 传输使用，不改变 Host 地址的 CPU 可用性。

## 固定区与流式窗口

扣除必要权重、Context、workspace、bridge 和索引后，剩余 expert 容量共同规划固定区与窗口。
固定集合按各层优先级和均衡字节额度选取，数量与地址在缓存寿命内稳定。

大 T 的一个窗口只需容纳本层固定副本之外的缺失 experts；窗口容量与固定额度联合求解，
不先永久扣除两整层 bank。通常用两个窗口交替加载和预取，容量不足时可使用一个。
小 T 把窗口空间合并成额外缓存，继续扩展每层优先级前缀。

| 实际 token 规模 | 执行方式 |
|---|---|
| 小 T 稳定状态 | GPU 计算 ready 子集，CPU 计算补集；复用驻留快照 |
| 大 T | 当前窗口补齐整层，全部 GPU 计算；另一窗口预取下一层 |
| 大 T 返回小 T | 仅重建窗口额外集合，已 ready 部分立即可用，剩余部分由 CPU 计算 |

分派依据实际 token 数与计算 profile，Prefill、Decode、Verify 和 Control 使用同一机制。
切换不搬动固定区。不同层的编码大小和整 expert 取整会留下有限尾部余量，由真实布局核算。

静态优先级只决定副本集合，不改变路由。默认标定属于模型，显式排序可以覆盖；
当前机制不在每个请求中学习并动态替换 expert 集合。

## 上传、借用与复用

```text
编码 payload 上传 → 索引发布 → consumer 等待就绪 → GPU 读取 → consumer 完成 → 窗口重用
```

上传按真实连续性合并或批量提交，使用有界描述符和在途槽位；一个 expert 可以跨批，但必须在
完整 payload 已有序提交后发布。Pinned 描述符和索引快照保持到对应传输完成。

窗口重用等待旧消费者。Host release 只表示已经提交最后消费者的完成依赖，不表示 GPU 已经
停止读取。新的预取不能成为旧 ready 数据无关的等待条件，也不能挡住当前层尚未发布的数据。

缓存 acquire/release 与布局变更在 capture 外管理；Graph 消费稳定索引与相应代次。
已经提交的 DMA 不能靠撤销 Host 请求取消，失败和析构先结束真实消费者再释放来源。

## CPU/GPU 交接

混合执行提交 route，回传所需 hidden 和路由，再让 CPU collective 与 GPU routed/shared 重叠。
CPU 输出 FP32 分项，传回后与 GPU 结果按最终舍入合同合并。Bridge 和上传权重使用各自有寿命的
缓冲与依赖，不能在上一层的 Host 分项仍被读取时覆写。

CPU team 是通用执行基础，不调用 CUDA；RankProgram 决定何时输入就绪、何时运行和合并。
实际没有 CPU assignments 时跳过算术和分项回传。持续轮询只在需要的 traversal 中维持，
空闲请求不保持团队活跃。

稳定驻留可以合并连续 GPU 段；流式和恢复保留上传交接，CPU 工作使用分段图的就绪事件。
Eager 与 Graph 使用相同分工快照和权重寿命。

## 预算与观测

Host 来源包含已缓存到 GPU 的副本，驱逐无需写回。GPU ready 与 target 是 cache 容量中的状态，
注册范围是 Host backing 的子集，均不能作为额外分配重复相加。
资源计划同时覆盖容量、传输槽、CPU scratch 和结束路径，遵循[启动预算](engine-architecture.md#启动与资源准备)。

物理实现与策略入口分别见 [Expert cache](../../src/core/expert_cache.h)、
[MoE schedule](../../src/models/common/moe/schedule.h)和
[offload](../../src/models/common/moe/offload/)。
