# Tests

测试保护公开行为、数学、状态和生命周期。具体 Op 的数值方法见
[Op 开发](../docs/maintainer/op-development.md#数值验证)，跨模块所有权见
[Engine 架构](../docs/maintainer/engine-architecture.md)。

## 组织

测试按可观察风险组织，不镜像每个源文件。Op suite 持有独立参考与有限 case；模型测试检查
原生计算组合和状态事务；runtime 检查调度、缓存与资源；产品测试检查协议及统计消费者。
Artifact/converter 检查表示、绑定和跨语言读写。实际注册由各目录 CMake 维护。

Op case 是一个可复现输入或完整 state/Graph 序列。Case 声明和筛选先于设备初始化及大 fixture；
相关案例可共享不可变权重。独立 oracle 不按执行 target 拆成不同数学标准。
真实仅属于某 target 的能力由对应 suite 组织，普通 case 不因调优差异复制。

共享 guards、上传与比较在 `ops/` 测试支持中；criterion 和参考留在语义 suite。
Private selector、文件名、launcher 和代码结构不作为验收对象。

## 构建与筛选

在选定 Python 环境中准备所需测试依赖，使用开发 preset：

```bash
cmake --preset rtx5090-dev
cmake --build build/rtx5090-dev -j
ctest --test-dir build/rtx5090-dev -N
ctest --test-dir build/rtx5090-dev -R 'ninfer_context_capacity' --output-on-failure
```

Spark 使用 `dgx-spark-dev`；`*-ops` 注册对应基础与 Op suites。
Python interpreter 在本地 CMake preset 中显式选择。完整目标与链接规则见
[构建系统](../docs/maintainer/build-system.md)。

Op runner 支持元数据列举与交集筛选，例如：

```bash
./build/rtx5090-dev/tests/ninfer_linear_projection_test --list --format q4
./build/rtx5090-dev/tests/ninfer_linear_projection_test --format q4 --policy a16
```

用 `--help` 查当前 filters，`--case` 选择稳定 case ID；一个 Graph/state 序列整体执行。
测试默认集合覆盖实际生产风险，不把每次调优扫描变成长驻 case。增加检查应对应可能改变
结果或破坏状态的边界；不为覆盖计数穷举所有配置组合。

## 真实模型与分布式

真实模型检查使用明确 artifact，按所属入口提供环境变量或参数；不自动下载权重。
GPU integration tests 串行运行，以免互相争用资源。

- [Qwen3.5](models/qwen3_5/tests.cmake)：加载、生成、MTP/DFlash、前缀与压力恢复。
- [Qwen4Exp](models/qwen4_exp/tests.cmake)：加载、PLE、原生事务与模型组合。
- [GLM5Next](models/glm5next/tests.cmake)：资源、Frontend、并行、Main/MTP/Vision 与图执行。
- [分布式](distributed/)：执行组、消息、全 rank 完成和故障清理。

启用 `NINFER_ENABLE_DISTRIBUTED` 后可运行独立的执行组测试。部署 TOML 使用
`ninfer.example.toml` 的结构，将 node worker 指向各机对应测试 executable：

```bash
./build/dgx-spark-dev/tests/ninfer_distributed_execution_test --config distributed.local.toml
```

这个入口使用生产 transport、Executor 与 NCCL，小型 Program 提供独立计算参考；
它不证明某个真实模型已经正确集成。模型验证通过现有原生检查与公共 Engine/产品入口完成。

## 验证边界

浮点模型的不同合法路线可能产生不同 logits/token；数学正确性由 Op oracle 验证。
模型检查关注组合、前沿、采用前缀、缓存内容、取消和资源寿命，不以输出看似正常替代这些事实。

Prefix reuse 不保证任意负载序列后都命中。缓存测试需固定来源、保留资格与资源前提，
然后检查仍可恢复的内容；不能通过改变 prompt 或增加预热追求断言通过。

失败注入仅在相关测试 executable 内使用现有 interpose 支持，生产代码不增加测试 hook。
性能只由相应 benchmark/产品测量证明；确定性工作量检查可防算法复杂度回退，不能替代时延测量。
临时实验和日志使用 `.agent-tmp/<task>/`。
