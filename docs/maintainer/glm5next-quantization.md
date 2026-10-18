# GLM5Next 量化与数值表示

本文长期维护 GLM-5.3-Flash 的源权重表示、选定量化范围、精确编码、计算精度与
可观察舍入边界。[模型语义](glm5next-model.md)维护完整数学和逻辑权重角色。
这些是已确定的目标政策，不表示 converter、Ops 或模型接入已完成；实现状态见
[设计索引](../designs/glm5next/README.md)。

通用编码和字节布局分别由[数值格式](tensor-formats.md)、[存储布局](storage-layouts.md)
维护。已有格式直接引用；本文补充 GLM 的格式选择、源导入以及尚待实现的明确编码规则。
物理 parent、page、record 和 workspace 不在本文重复定义。

## 1. 权重来源与量化范围

两台 Spark 的来源路径都是：

```text
/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4
```

以这份 checkpoint 为唯一权重起点，只增加 KDA 大投影与 MTP routed experts 两项量化：

| 部分 | 目标持久表示 | 计算许可或保留内容 |
|---|---|---|
| Main routed experts | 原 NVIDIA NVFP4 | `AllowA4`；保留 codes、G16 E4M3 scales 和原校准参数的含义 |
| Main 前三层 dense MLP | 原 NVIDIA NVFP4 | `AllowA4`，与 routed experts 独立的 input/weight scales |
| 34 层 KDA 的 Q/K/V/O | `q8_g32_fp16`，由源 BF16 转换 | `A16Only`；只量化这四个大投影 |
| KDA beta、两组低秩 gate 投影、output norm | 原 BF16 | 投影 `A16Only`，不额外量化 |
| KDA convolution、A_log、dt_bias | 原 FP32 | 保留源 words 与控制含义 |
| Main/MTP MLA 与 Indexer 权重 | 原 BF16 | 投影保持 A16；Indexer head coefficients、点积累加和 score 归约保持 FP32 |
| Main/MTP shared experts | 原 BF16 | `A16Only` |
| Main/MTP router | 原 BF16 weight、FP32 correction bias | FP32 logits 与原 route 数学 |
| mHC、Text/MTP norms | 原 BF16 | mHC 系数、norm、混合的 FP32 精度及 BF16 residual 边界见第 5 节；其他 norm 按模型数学和第 4 节 |
| MTP routed experts | `fp8_e4m3fn_block128x128_bf16`，由源 BF16 转换 | `AllowA8` |
| MTP stem、embedding、LM head、全部 Vision | 原 BF16 | 保留源权重，投影 `A16Only` |

Cache 与计算政策及精确 codec 见第 3 节，与上表的静态权重量化分开。Main MLA Q_b/O、shared experts 不加入额外量化范围。

## 2. 静态权重编码与导入

### 2.1 NVIDIA NVFP4 导入

源 E2M1 codes 沿 K 两个值打包为一个 U8；每 K16 group 一个 E4M3 scale，每矩阵一个
FP32 `weight_scale_2`，每输入一个 FP32 `input_scale`。两种 scalar 都是 multiplier：

```text
w_source = E2M1(code) * E4M3(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
w_ninfer = E2M1(code) * E4M3(scale) / d_w
```

沿用 NInfer 的 divisor 格式时，codes/scales 可保值重排；FP32 reciprocal 本身有舍入，
不是逐位恒等变换。`d_a` 属于使用位置的 activation metadata，不能与 weight divisor
合并解释。A4 输入的 K16 量化及 clamped SwiGLU 的激活出口按相应 Op 合同处理。

已核对该源 checkpoint 的全部 42×288 个 Main experts：同 expert 的 gate/up `weight_scale_2`
相同；同层所有 experts 的 gate/up `input_scale` 相同，down `input_scale` 也各自
相同。前三层 dense MLP 的 gate/up 两类 scalar 也分别一致。因此 GU 可以在不修改
这些校准值的情况下沿输出行合并。

### 2.2 KDA Q8 与 MTP block-FP8

KDA Q/K/V/O 沿 K 每 32 个值使用一个 FP16 scale，转换采用现有 `grouped_absmax`
数值规则；每个逻辑权重占 1 B code＋`2/32` B scale。输入 activation 保持 A16，
量化后的权重进入既定 KDA 公式，不改变 convolution、gate 或 recurrent state 类型。

MTP 的 GU/down 使用 128×128 块 E4M3FN codes 与 BF16 multiplier：

```text
w_hat[e,n,k] = FP32(E4M3FN(code[e,n,k]) * BF16(scale[e,n/128,k/128]))
```

每个 expert 的 gate/up/down 分别量化，scale 不跨 expert 或矩阵边界。GU 沿 N 合并时，
2048 行的分界与 128 行块对齐。权重没有静态 activation divisor；允许的 A8 路径按
token 的 K128 分组动态量化激活，采用各自 activation scales。

BF16→block-FP8 的转换方法定为 `fp8_block_maxabs`，沿用
[现有行 FP8 编码器](../../tools/convert/quantization/fp8_row.py)的数值规则，将 max-abs
归约范围改为每个独立的 128×128 块。源 BF16 words 精确展开为 FP32；全零块写
正零 BF16 scale 和全正零 E4M3FN codes。非零块按以下顺序编码：

```text
amax       = max(abs(block_fp32))
raw_scale  = RN_FP32(amax / 448)
scale_word = RN_BF16(raw_scale)
if scale_word == 0x0000:
    scale_word = 0x0001                    # 最小正 BF16 subnormal
scale32    = exact_bf16_to_fp32(scale_word)
inv_scale  = RN_FP32(1 / scale32)
normalized = RN_FP32(weight_fp32 * inv_scale)
code       = RN_E4M3FN(clamp(normalized, -448, 448))
```

所有 `RN` 均采用 round-to-nearest、ties-to-even。Codes 使用**实际存储的 BF16
scale** 的 FP32 reciprocal 编码；保留先算 reciprocal、再乘 weight 的舍入顺序。
非零块内的零值保留符号。源含 NaN/Inf、非零块 scale 非正或非有限、归一化结果
非有限时，转换报错。流式读取批次和后续 GU 合并均不改变块的归约域及编码结果。

这一方法仅使用源权重，无需激活校准数据；运行时动态 A8 量化保持前述 K128 约定。
编码表示和 permission 的语义不随 converter 的实现进度变化。

## 3. MLA／Indexer cache 与计算政策

| 部分 | 持久表示 | 计算 |
|---|---|---|
| MLA latent | BF16 或 E4M3FN/G128/FP32 scales；原始基底 | BF16 操作数、FP32 累加；softmax 指数、统计和分块归约 FP32 |
| Indexer pooled keys | 固定 BF16，无 scale/Hadamard | BF16 Q/K、FP32 累加；head coefficients 和评分归约 FP32 |
| Indexer raw tail/records | BF16 K/gate，不含 APE | Pooling softmax/weighted sum FP32，pool 出口 BF16 |

Main/MTP 和两 ranks 在启动时使用同一个 MLA storage profile，运行中不切换。
本 profile 的 16-bit 操作数是 BF16，不引入 FP16、FP8 query 或 FP8 点积分支。
FP8 只改变 latent 存储；格式选择不授予更低精度计算许可。

Indexer 固定 BF16 cache/计算。只有后续实测成为瓶颈并由用户明确重新决定，才可
考虑 FP8 cache＋BF16 计算；这一条件也不自动授权 FP8 query 或 FP8 点积。
执行路线不得按 T/B、显存压力或设备事实修改这些精度政策。

### 3.1 两个存储 profile

- `BFloat16`：每 token 保存一个 BF16 `c[512]`，1024 B，无 scale。
- `Fp8E4m3fnGroup128F32`：512 个 E4M3FN codes、4 个 FP32 scales，528 B/token。

两种 profile 都在原始 latent 基底中，无 Hadamard 或其他旋转。只有一份 latent，
供 K/V 两个角色共用；不分别存储展开的 K/V。不能套用现有 GQA 的 BF16-K/FP16-V
解释或 D256 cache codec。

FP8 按每 token 的连续 128 维分成四组。四 token 是加载单位，不是量化组：不跨
token 共享 scale，不因新 token 到来重新编码旧 latent。

### 3.2 编码

输入为 latent RMSNorm 实际产生的 FP32 x；BF16 profile 直接写 `RN_BF16(x)`。
FP8 profile 直接编码 FP32 norm 结果，不增加中间 BF16 舍入。

对每个 128 维组：

```text
a = max_i(abs(x[i]))

if a == 0:
    scale   = FP32(+0)
    code[i] = E4M3FN(+0)
else:
    scale   = max(RN_FP32(a / 448), 2^-126)
    inv     = RN_FP32(1 / scale)
    z[i]    = RN_FP32(x[i] * inv)
    code[i] = E4M3FN_RNE(clamp(z[i], -448, 448))
```

所有 RN 为 round-to-nearest、ties-to-even；scale 为实际存储的 FP32 multiplier，
不取整到二次幂。全零组规范化为正零；非零组中的 signed zero 按上述编码保留。
按所列 FP32 步骤执行，不能用另一种除法/scale 舍入规则替换而改变 code words；
也不能通过 flush-to-zero 改变规定的极小值行为。

有效输入、需要的中间值及计算结果必须有限；NaN/Inf 不属于有效计算域，不能靠
satfinite 掩盖。Device 内容的有效性由调用者保证，不引入 Host readback 扫描。

### 3.3 消费

```text
decoded[i] = RN_FP32(FP32(E4M3FN(code[i])) * scale[group(i)])
operand[i] = RN_BF16(decoded[i])
```

解码在片上完成，QK 与 latent 加权累加使用同一 BF16 operand。没有全局展开的
BF16 cache。新写入 token 也必须经过所选 profile 的编码/解码边界，不能用未编码
的当前窗口 latent 绕过 FP8 精度。融合可以消除内存往返，不能消除这个表示边界。

Storage codec 的精确规则不要求完整 projection/norm 的不同合法实现逐位相同；
codec 精确验收针对给定 FP32 输入，完整 prepare 另按独立数学 oracle 验证。

### 3.4 与上游 Indexer 表示的区别

本地 vLLM 参考路径会对 query 和 pooled key 做归一化 Hadamard-128，再以 E4M3FN
per-vector 编码，使用取为二次幂的 FP32 scale。NInfer 不采用这套 Indexer 表示。
Transformers 直接参考路径也没有该 codec；后端中间舍入与 NInfer 的政策分别解释，
不能把某个后端输出当作独立数学 oracle。此差异独立于 NVIDIA 的静态权重量化。

## 4. 投影、控制量与 latent attention 边界

Norm 使用模型规定的 epsilon 和直接 gamma。Text/Vision RMSNorm 为 `1e-5`，
KDA Q/K L2 norm 与 Indexer LayerNorm 为 `1e-6`；归约和 norm 算术为 FP32。
不采用 Qwen4Exp 的 offset-gamma。所有源 FP32 convolution/control/bias 保留其 words。

### 4.1 KDA 与 FFN

KDA 输入保持 BF16 raw forget/beta logits，内部以 FP32 A_log/dt_bias 生成实际
log-decay/beta；不把激活后的 FP32 gate 改成外部投影输入。运行和发布的 recurrent
state 为 FP32，卷积历史为 BF16。Replay 对本轮实际 gate words 的精确记录和重放
属于 record/fold 合同，不改变上述计算输入。

Router 使用 FP32 logits、sigmoid scores、route coefficients 及原 correction bias
语义。NVFP4、Q8 与 block-FP8 各自的 activation permission 见第 1 节；量化不得
移动 SwiGLU clamp、route normalization 或 shared contribution 的数学位置。
源权重导入与 GU 合并必须保持 gate/up 各自的量化域和校准含义，不能跨 expert
重新选择 scale。具体 parent 与 TP/EP 局部几何由实现设计定义。

### 4.2 MLA／Indexer 输入准备

同输入的准备投影使用 BF16 操作数、FP32 累加。其五类结果的表示边界为：

| 逻辑结果 | 数值边界 |
|---|---|
| Q_a | FP32 RMSNorm 后输出 BF16 q_r |
| KV_a | FP32 RMSNorm 后直接进入第 3 节 encoder，不先转 BF16 |
| Index K | FP32 affine LayerNorm 后输出 BF16 raw K |
| Pool gate | 投影结果转 BF16 raw gate；APE 尚未加入 |
| Index head coefficients | 保留 FP32，不做 BF16 舍入；公开 raw coefficient 不含 `1/sqrt(32)` |

RMSNorm 为 `x*gamma/sqrt(mean(x*x)+1e-5)`，LayerNorm 为
`(x-mean(x))*gamma/sqrt(mean((x-mean(x))^2)+1e-6)+bias`；epsilon 按 FP32 常量解释。
准备投影在 norm 前没有额外 BF16 落点。Pooling、tail 和 records 消费同一份实际
BF16 raw K/gate；即使完整四个 tokens 都在当前调用内，也不绕过这个边界。
APE 在 pooling 时以 FP32 加入，每个 channel 沿四 token 做 softmax 和 weighted sum，
pool 出口转 BF16，无额外 norm、旋转或量化。

### 4.3 Latent 域 attention

本模型保留 BF16 U_K/U_V 因子，不离线与 Q_b/O 合成扩大后的吸收矩阵。
QK、加权累加和因子投影保持 BF16/F32 计算约束；softmax max、sum、exp 与分块合并
为 FP32。q_latent/u 是私有中间值，不强制全局物化，也不为复刻拆分调用增加额外
BF16 落点。允许保留 FP32 中间量以融合计算，不能将矩阵操作数降低到 FP8。
公共输出是完整 U_V 映射后的 BF16 heads，完整数学见模型第 4.1 节。

cache_operand 按第 3.3 节产生，BF16 存储则直接使用存储值。FP8 的当前 token
同样经过实际 encoder/decoder，不从未编码的临时 latent 走旁路。Attention scale
固定为 1/16，不因 512 维 latent 改成 `1/sqrt(512)`。O projection 消费 BF16 heads。
空选择集合的写域和精确零规则由消费者接口规定。

## 5. mHC 数值边界

### 5.1 当前固定 profile

| 项目 | 约定 |
|---|---|
| `D / S` | 4096 / 4 |
| `eps_coeff / eps_branch / eps_final` | `1e-5` |
| `eps_pre / eps_sinkhorn` | `1e-6` |
| `sinkhorn_iters` | 20 |
| post multiplier | 2 |
| `X / X_next / embedding / u / y / hidden` | BF16 |
| 源与驻留 `F / base / scale / gamma / gamma_final` | BF16，保留源 words |
| `F*X`、平方和、collapse 与 residual 混合 | FP32 累加／计算 |
| logits、sigmoid、softmax、Sinkhorn、norm | FP32 |
| 跨分支保存的 `q / C` | FP32 |

配置中的几何与 epsilon 属于模型语义；上表是已选定的 profile，不能把这些参数
当作调优旋钮。不同 token 数和私有执行路线保持同一精度政策，不增加 FP8 residual、
低比特 mHC 权重或低精度动态系数分支。

原始 `F` 和 `X` 都是 BF16，可以使用 BF16 乘积、FP32 累加完成 `F*X`，再以 FP32
应用 `r`。这不要求把整个 `F` 常驻展开为 FP32。小型 BF16 参数读取后以 FP32
参与运算；checkpoint 中 BF16 表示和执行时 FP32 计算是不同概念。

### 5.2 必须保留的 BF16 residual 边界

每次 post 的 BF16 residual 是下一 site 的实际输入。融合 `post_pre` 必须先形成
这些 BF16 值，再计算下一 site 的投影、全宽 RMS 统计及 collapse。

```text
X_next_math → RN_BF16 → X_next
                         ├→ next F projection / full-width RMS
                         └→ next pre collapse / branch RMSNorm
```

这里的转换可以发生在寄存器或 shared memory，不要求额外显存往返；但不能一面
向外写 BF16 `X_next`，一面用未舍入的 `X_next_math` 生成下一组系数。
`RN_BF16` 使用 round-to-nearest、ties-to-even；归约结合顺序和执行分解不由本合同固定。

### 5.3 不增加的内部低精度舍入

以下值不是独立 BF16 出口：全宽归一化向量、24 行 logits、`p`、collapse `v`、
单独的 `C^T X` 或 `q*y`、末尾四流均值。它们按本 profile 的 FP32 精度继续计算。

特别是：

- `collapse → branch RMSNorm` 之间不先转 BF16。
- 最后 `mean → final RMSNorm` 之间不先转 BF16。
- `C^T X` 与 `q*y` 相加之后才产生 BF16 residual，不能分别写 BF16 再相加。
- 公共 `q/C` 是 FP32，混合前不降为 BF16。

本合同不要求与某个参考后端逐 bit 相同。已核对的 vLLM revision
`00b7847c8036b667742b4efb21aab1de51fd4721` 中，[独立 pre](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/model_executor/kernels/mhc/torch.py)
读取 BF16 residual，而一个 [mhc_fused_tilelang 路径](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/model_executor/kernels/mhc/tilelang_kernels.py)
使用未舍入的 `new_r` 生成下一组投影和平方和，同时写 BF16 residual。
其 pre/norm 融合路径也存在 BF16 staging。
这些实现细节不能成为 NInfer 不同路线采用不同数学输入的依据。

因此，参考实现用于核对模型数学和参数映射；NInfer 使用本节统一的表示边界与
独立数学 oracle 验收。保留表示边界与删除额外内部舍入都应覆盖在融合测试中。

初始化四流 residual 按 embedding 的 BF16 words 精确复制，包括 signed zero。
最终 post 同样保留 BF16 residual 边界；随后 mean/final norm 以 FP32 完成，只在
最终 hidden 出口转 BF16。跨分支的 q/C 是实际 FP32 出口，后续混合直接消费它们。

## 6. 权重容量

源容量由已核对 checkpoint 的 33 个 safetensors 文件头逐 tensor 统计，共 147661 个 tensors；
与源 index 的 `total_size=204419110596` B 一致；两台来源的模型/量化配置相同。
下表是全模型的 tensor payload，单位 GiB；不含文件头、对象对齐、runtime、分片复制
或额外模型权重。目标列是按已选格式计算的估算，尚非实际 artifact 大小。

| 部分 | NVIDIA 来源 | 选定表示后的估算 |
|---|---:|---:|
| Main routed experts | 159.469020 | 159.469020 |
| Main dense MLP | 0.237305 | 0.237305 |
| Main KDA | 8.729325 | 4.744950 |
| Main MLA＋Indexer | 2.559384 | 2.559384 |
| Main shared experts | 1.968750 | 1.968750 |
| Main router | 0.092330 | 0.092330 |
| Main mHC＋norm | 0.066617 | 0.066617 |
| Embedding＋LM head | 2.363281 | 2.363281 |
| MTP routed experts | 13.500000 | 6.750824 |
| MTP 其他独有权重 | 0.344283 | 0.344283 |
| Vision | 1.049837 | 1.049837 |
| **合计** | **190.380132** | **179.646581** |

KDA 的 Q/K/V/O 原载荷恰为 8.5 GiB，Q8-G32 后 4.515625 GiB，减少 3.984375 GiB。
MTP experts 共 7247757312 个权重，block-FP8 codes＋BF16 scales 为
7248642048 B，减少 6.749176 GiB。两项共减少约 **10.733551 GiB**。

不启用 MTP 时应扣除其 experts 与其他独有权重；embedding/head 不重复扣除。
不启用 Vision 时扣除独立 Vision 权重。双机每 rank 的权重容量还取决于实际分片与复制
计划，不能直接以总量除二作为最终驻留预算。

NVIDIA 模型卡中量化 shared experts 的措辞与实际配置不符，所链接 recipe 中「MTP
未导出」的注释也不符合这份本地 checkpoint。本文采用实际 tensors：shared 为 BF16，
完整 MTP 权重存在且原 experts 为 BF16。

## 7. 来源与通用合同

源 checkpoint、配置和模型实现入口见[模型来源](glm5next-model.md#8-来源)。
NVFP4 源表示同时参照
[NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)，
实际存在的 tensors 和容量以源文件核对为准。

Q8 的 grouped_absmax、NVFP4 divisor 与已有 block-FP8 represented values 遵循
[通用数值格式](tensor-formats.md)。实际格式已有定义时不另设 GLM 专属解释；尚待
实现的编码规则在代码／通用格式接管后改为直接引用，模型的格式选择长期保留。
模型加载、转换命令和产物状态不由本参考文档宣称，见设计索引及实际 converter。
