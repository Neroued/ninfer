# Kpool Sparse MLA 交付与后续清单

本清单区分 BF16 实现交付、逐点性能验收和模型集成。BF16 Op family 已实现；
本次整理用于迁入 dev，不代表所有负载已达到 roofline，也不代表 GLM Engine 已接入。
数学与读写合同以 [kpool_mla.h](../../../include/ninfer/ops/kpool_mla.h) 为准，
布局以 [kpool_mla_layout.h](../../../include/ninfer/ops/kpool_mla_layout.h) 为准。
权重绑定、模型调用链和资源交接见 [family 设计](kpool-sparse-mla.md)。

## BF16 已交付范围

- [x] 无地址 profile、缓存／tail／records／selection views、布局 planner 和绑定。
  Native 页组包含全部层；同一页表通过不同 planes 供 Prepare、Indexer 和 MLA 使用。
- [x] Prepare 的 Full、QueryAndHistory、History 三种投影需求，以及 record／update
  两种状态出口；FP32 投影接 norm，实际 BF16 raw words 接 learned Kpool4 pooling。
- [x] 跨层 tail extraction；原位／异位目标、部分接受和零接受的精确写域。
- [x] BF16 weighted Indexer 和精确 Top512；负系数、负分、同分次序、全选边界、
  1M 上下文、多级归并及有界 score workspace。
- [x] 完整 BF16 latent sparse MLA：U_K → latent attention → U_V；直接读取 Kpool4，
  frozen selection 与 incomplete tail、空集合和无效行、split 与 merge/value 融合。
- [x] 四个容量查询与执行共用分配规则，覆盖 T 区间高水位；执行不分配 backing，
  不重打包权重，不发布逻辑 frontier。
- [x] 固定 T/B 时支持 Device metadata 更新；Indexer／MLA 的 Host envelope 改变经
  实际 Graph 更新生效，节点数量和顺序保持兼容。
- [x] 同流 Prepare → BF16 W_B split → Indexer → sparse MLA 的独立数学组合验证；
  W_B 前缀和 Q8 A16 O 使用 Linear 已有公开入口。
- [x] 单 Op、完整 attention 分支、上下文增长、ragged batch、共享前缀和 MTP 操作序列
  的公共 benchmark；测量范围是 Op 组合，不是 Engine TPOT。
- [x] 各 Op 私有 selector、launch bindings 和计算体分开；只共享 Top512 计算体，
  QSA 的评分和调优政策仍归 QSA。临时候选和强制分派入口不进入主线。

实现支持 DGX Spark 上的 BF16 Prepare／Indexer／sparse MLA。FP8 latent 的布局可被
规划，Indexer 可访问其中的 BF16 index plane；FP8 Prepare 和 sparse MLA 尚未实现，
容量查询和执行会拒绝，不能因 enum 或 layout 存在推断执行支持。

## 资格与复现

测试入口、筛选和执行方法见 [tests/README.md](../../../tests/README.md)，测量命令、
JSON 配置和统计口径见 [bench/README.md](../../../bench/README.md#kpool-mla)。
所有资格测试调用公共 Op，有限案例覆盖生产分派边界、窗口尾部、页搬迁、Graph 更新、
selection 复用、实际写域及 workspace。Prepare／MLA 使用独立 FP64 数学参考；
Indexer 将精确选择与评分舍入误差分开，tail 检查实际 words 的精确复制。

数值通过不代替性能验收。小 T 检查必要流量与持续带宽，大 T 同时检查计算上限、
权重复用、KV 局部性和完整入口开销。硬件参考只取
[bench/common/roofline.h](../../../bench/common/roofline.h)，不在 family 中复制校准常数。

## 仍待完成的性能工作

- [ ] 完整 BF16 分支的每个目标点达到对应 roofline。既有应用测量中，长上下文大 T
  和部分 ragged batch 仍有明显差距，不能勾为性能完成。
- [ ] 长上下文 prefill 的 sparse MLA：改善分散 Kpool4 读取、跨 query 的实际 KV
  复用和因子映射开销。只在 selection 确实共享时复用，不推断相邻 query 的集合相同。
- [ ] 长上下文多 batch Indexer：在已保留的按请求集中处理策略上，继续分析 key 重扫、
  score/candidate 访问和 Top512 成本；完整入口收益才可决定新分派。
- [ ] 分别评估 fresh prefill 和带长前缀的小 chunk。展开 K/V 的方案须同时计入
  展开计算、额外显存、带宽和可见集合复用，不能只依据较短的 attention 收缩维度。

令每个 query 实际选中 token 数为 S，候选 pool 数为 C；主要工作量为：

| 部分 | Useful FLOPs | 关键必要 payload |
|---|---:|---|
| Prepare | `2*4096*R_A*M_live` | W_A 18.25／18／6 MiB，三种投影需求分别计费 |
| Indexer score | `2*32*128*Σ(C>512 ? C : 0)` | 每个不同历史的 key 并集为 `256*C` B |
| U_K＋U_V | 通常 `4*64*256*512*M_live` | 两份因子共 32 MiB；空／单 token 集合按有效数学计费 |
| Latent QK＋PV | `4*64*512*ΣS` | 每 query `1024*S` B，另报不同 query 的真实地址并集 |

完整分支的外部流量下界排除可能在片上连续消费的中间结果；各阶段实际 scratch 和
重复读取另行分析。`max(bytes/BW, FLOPs/compute)` 是全局理想下界，不能解释成已经
证明可达的融合调度。L2 请求量不等于 DRAM 流量，事件插桩阶段的中位数也不能相加
代替完整 Graph body 时间。验收只对照 roofline，不以旧实现提升百分比替代。

开发证据保留在本地 `.agent-tmp/glm5next-mla/`，使用 Markdown 和图表；不把候选源码、
性能日志或 profiler 产物提交到主线。新性能决策须在最终公共入口、相同负载与测量条件
下复测，并记录仍无法解释的离散性。

## FP8 latent 后续

- [ ] 实现 G128／FP32-scale encoder，直接消费 FP32 norm；遵守
  [已定 codec](../../maintainer/glm5next-quantization.md#3-mlaindexer-cache-与计算政策)。
- [ ] Sparse MLA 片上解码为 BF16 并在 QK/PV 间复用；新 token 经过同一表示边界，
  不创建全局 BF16 展开缓存。
- [ ] 独立 codec 向量和完整数学资格检查；重新测量含量化、scale 读取和解码的完整入口。
  FP8 与 BF16 的调优政策及 roofline 分别维护。

## 模型接入后续

- [ ] Converter／binding 物化 W_A 前缀、W_B split、U_K/U_V 和 Q8 O；执行不做 packing。
- [ ] RankProgram 通过 planner 规划 Main 11 层和 MTP 1 层 history、tail、frame 和
  workspace；通过所属 Op 查询能力，不复制 kernel 支持条件。
- [ ] Session／目录接入 COW、page/state leases、接受／拒绝、checkpoint 和覆盖范围。
- [ ] 模型执行接入 MTP Full 首步、QueryAndHistory selection 复用、History canonical
  补齐，以及 record → 接受前缀 → tail extraction → 下一轮的完整事务。
- [ ] 模型 Graph 管理稳定地址、envelope 容量档位、更新和资源退役；跨流依赖由资源所有者建立。
- [ ] ADP owner、通信、mHC／residual 与公共 Engine 路径完成集成后，单独验收模型数值、
  生命周期和双 Spark 吞吐；Op benchmark 不替代这些证据。
