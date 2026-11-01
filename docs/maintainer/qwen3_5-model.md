# Qwen3.5 架构

Qwen3.5 Dense 与 MoE 共用混合 attention 架构，FFN 使用各自数学实现。
Qwen3.6/3.8 的已注册 27B 和 35B-A3B 是它们的实例；名称与量化组合不创建新的执行入口。
配置和绑定分别由 [config](../../src/models/qwen3_5/config.cpp)与
[load](../../src/models/qwen3_5/load/)维护。

## 计算结构

```text
embedding / Vision features
  → 多层 pre-norm residual：GQA 或 GDN → Dense FFN 或 MoE
  → final norm → Main head
```

Config 决定层序、head、宽度和组件关系，原生代码决定固定数学。Text 使用 offset RMSNorm，
gain 为 `1+w`；不能在转换时先把它舍入成普通 BF16 gain。完整 GQA 带输出 gate，
Q/K 使用 head norm 和部分 interleaved MRoPE。GDN 使用因果卷积、Q/K normalization、
gated delta recurrence 和 gated norm，状态按层/head 独立。

Dense 使用 SwiGLU；MoE 使用全局 top-k softmax routed experts 与独立 sigmoid gate 的 shared
expert。表示选择不改变 route 或 shared 的数学位置。具体公式、alias 与舍入合同在
[Ops](../../include/ninfer/ops/)，模型只负责层组合与实际参数。

## 表示与执行

Converter 可以为不同逻辑角色选择 groupwise INT、NVFP4、BF16 等表示。
Activation permission 属于使用位置；实际 Op 负责其格式和几何支持。Norm、卷积与逻辑行序
由 converter/load 对齐，运行期不重打包。通用关系见[权重与加载](weights-and-loading.md)。

本地执行使用原生 Session 和 RankProgram。27B Dense 与 35B MoE 保持各自 FFN 组合，
MoE 可以使用 GPU resident 或 RTX 5090 的[混合执行](expert-weight-cache.md)。
Speculative backend、proposal head 和 Vision 在启动选择，未选择的组件不占用执行资源。

## State 与历史

完整续算状态包含 GDN recurrent/conv、continuation hidden 和所选 draft backend 的定长状态。
Main GQA、MTP 和需要 full context 的 DFlash 层分别有自己的 history。
DFlash local cyclic KV 属于 State；临时 query KV、verify records 与候选分布属于 frame。

Checkpoint 保留同一次真实执行的 State、history coverage 与位置身份。
Prefill/Replay 可在一个 Work 内推进多个数学或媒体段，保存与调度边界由 Session 统一决定。
存储、引用、Host 恢复与抢占复用[Context 管理](context-management.md)，不按模型另建缓存策略。

## Speculative backend

MTP 是一个私有 attention＋FFN block，conditioning 使用 Main 最终归一化 hidden，
输入为分别归一化后的 embedding 与 hidden 拼接。Main/MTP 稳定前沿为
`F / max(F-1,0)`；token、hidden 和位置按真实配对解释。媒体占位使用对应的 composed embedding。

DFlash/DFlash2 从指定 Main 层的 residual 捕获 conditioning，一次 masked block 提出多个 token。
DFlash2 还有动态卷积与条件候选路径。它们的 context append、proposal 分布与状态恢复和 MTP
不同，详见[Speculative 执行](speculative-decoding.md)。

Indexed proposal head 是独立可选表示，通过 row map 回到真实 token ID；target 仍使用完整
Main head。不同 proposal 只改变接受行为和成本，不改变 target 验证职责。

## Vision 与 Frontend

Qwen Vision 使用 patch projection、带位置编码的 transformer tower 和空间 merger。
其 LayerNorm、Q/K RoPE 与 GELU 形式由 Vision 数学维护，不套用 Text 的 offset norm。
共享 Qwen processor 产生 patches、媒体 spans 和三轴位置；Main 在 embedding 后替换媒体列。

Text 位置随媒体 grid 延伸，decode 保留相应位置偏移。媒体内容和位置共同参与前缀身份，
仅比较占位 token 不足以复用。预处理和 GPU 窗口采用[公共有界媒体机制](frontend.md#有界媒体准备)。

共享 Qwen Frontend 负责模板、thinking、工具格式和输出事务，产品协议不解析模型标签。
模型与前端的组合入口分别位于 [qwen3_5](../../src/models/qwen3_5/)和
[common/qwen](../../src/models/common/qwen/)。
