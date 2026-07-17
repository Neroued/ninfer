# Op 数值正确性审查与整改记录

> 日期：2026-07-17
>
> 范围：活跃数值文档、27B/35B Op 契约、已实现 kernel 与 Op oracle

当前规范以 [`op-development.md`](../op-development.md)、[`AGENTS.md`](../../AGENTS.md) 和活跃
Op/目标文档为准。本记录只总结本轮保留的结论。

## 1. 数值约定

- 每个浮点 Op 只有一个独立、朴素的 FP32/FP64 数学 oracle；exact transform/codec 使用独立
  exact oracle。
- oracle 从公开输入所表示的值出发；packed 权重按 signed code 与 exact stored scale 解码。
  production kernel、Python execution route、另一条 GPU 路径和 tolerance-specific expected
  value 都不是第二个 oracle。
- 公开输入输出 dtype、显式 Cast/quantize/dequantize、registered codec 和持久状态是语义边界。
- kernel 的 accumulator、instruction operand、staging cast、reduction association、workspace
  dtype、中间物化和 launch decomposition 是实现选择。oracle 的高精度不规定 kernel 的执行
  顺序。
- fused kernel 不必复刻旧的 BF16 物化结果，也不被禁止采用自然的低精度中间路径；每条
  production route 直接按自身 tolerance 与同一个 oracle 比较。
- 测试使用普通确定性输入和真实执行域，不构造 seam-sensitive/discriminator fixture。

## 2. 保留的实现调整

以下调整仅描述当前 fused 路线的自然实现，不构成其他路线必须遵守的 rounding 规范：

| Op | 当前实现调整 |
|---|---|
| `GdnGatingProj` | 已经位于同一 epilogue 中的 projection accumulator 直接参与 control 函数 |
| `LinearSwiGLU` | fused GEMV/MMA epilogue 直接计算 SwiGLU；原有 materialized 路线继续保留 |
| `LinearAdd` | 寄存器中已持有 accumulator 的 fused GEMV/SIMT/MMA epilogue 直接完成 residual add；materialized 路线和需要 shared-tile 重排的路线保留各自自然表示 |
| `VisionPosEmbedAdd` | 当前 warp/CTA/generic kernel 在同一表达式中完成 interpolation 与 add |

这些实现仍以高精度 oracle 和 route-specific tolerance 判断正确性。Attention/GQA 的 kernel、
workspace、reducer、测试和性能状态保持原设计。

## 3. Oracle 整改

- GDN、GDN projection、LinearSwiGLU、LinearAdd 与 VisionPos 使用直接 FP64 数学 oracle。
- AttnInputProj 与 GdnInputProj 从最终 packed payload 独立解码后执行 sampled FP64 dot。
- Embedding 从最终 Q6/W8 payload bytes 独立解码；exact transform/codec 继续使用 exact
  oracle。
- CausalConv、GatedDeltaRule、VisionAttention、RoPE 与 column extraction 的公开入口直接
  与 naive oracle 比较，不再以另一条 production 路径作为 expected result。

## 4. 35B 与 MoE

35B 文档定义公开 tensor/state dtype、逻辑公式和高精度 oracle，不规定 private routing
probability、expert activation、workspace 或 reduction 的物理精度。未来 `SparseMoeAdd`
kernel 可选择自然的内部数值路径，并按同一个完整 Op oracle 资格化。

Router softmax 只覆盖 routed expert rows `0..255`；row 256 是独立的 shared sigmoid score。
当前没有新增或修改 MoE CUDA kernel，也没有把 artifact-native Python execution route 当作
MoE 数学 oracle。
