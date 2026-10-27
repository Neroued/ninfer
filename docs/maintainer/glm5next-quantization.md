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

以这份 checkpoint 为唯一权重起点。KDA 大投影、Main/MTP MLA O 与 shared FFN
采用 Q8 A16，MTP routed experts 采用 block-FP8，另生成完整词表的 Q4 draft head：

| 部分 | 目标持久表示 | 计算许可或保留内容 |
|---|---|---|
| Main routed experts | 原 NVIDIA NVFP4 | `AllowA4`；保留 codes、G16 E4M3 scales 和原校准参数的含义 |
| Main 前三层 dense MLP | 原 NVIDIA NVFP4 | `AllowA4`，与 routed experts 独立的 input/weight scales |
| 34 层 KDA 的 Q/K/V/O | `q8_g32_fp16`，由源 BF16 转换 | `A16Only`；只量化这四个大投影 |
| KDA beta、两组低秩 gate 投影、output norm | 原 BF16 | 投影 `A16Only`，不额外量化 |
| KDA convolution、A_log、dt_bias | 原 FP32 | 保留源 words 与控制含义 |
| Main/MTP MLA O | `q8_g32_fp16`，由源 BF16 转换 | `A16Only`；仅量化 `o_proj` |
| Main/MTP MLA 其余权重与 Indexer | 原 BF16 | Q_a/Q_b/KV_a/U_K/U_V、Indexer 全部保留；投影保持 A16，Indexer head coefficients、点积累加和 score 归约保持 FP32 |
| Main/MTP shared experts 的 gate/up/down | `q8_g32_fp16`，由源 BF16 转换 | `A16Only`；GU 合并，down 独立 |
| Main/MTP router | 原 BF16 weight、FP32 correction bias | FP32 logits 与原 route 数学 |
| mHC、Text/MTP norms | 原 BF16 | mHC 系数、norm、混合的 FP32 精度及 BF16 residual 边界见第 5 节；其他 norm 按模型数学和第 4 节 |
| MTP routed experts | `fp8_e4m3fn_block128x128_bf16`，由源 BF16 转换 | `AllowA8` |
| MTP stem、embedding、Main LM head、全部 Vision | 原 BF16 | 保留源权重，投影 `A16Only` |
| Optimized draft head | `q4_g64_fp16`，直接由源 BF16 LM head 转换 | `A16Only`；完整词表，不裁剪；按配置加载并供 draft 使用 |

Main head 保持 BF16；未启用 optimized draft head 时，MTP 使用 Main head。
Cache 与计算政策及精确 codec 见第 3 节，与上表的静态权重量化分开。

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

### 2.2 Q8 投影与 MTP block-FP8

KDA Q/K/V/O、Main/MTP MLA O、shared gate/up/down 沿 K 每 32 个值使用一个
FP16 scale，转换采用现有 `grouped_absmax` 数值规则；每个逻辑权重占
1 B code＋`2/32` B scale。输入 activation 保持 A16，
不增加 A8/A4 激活量化。每行的 K 分组独立，GU 沿 N 合并保持各行 codes/scales；
TP 的 K 切点保持 G32 对齐。量化后的权重进入既定公式，不改变 KDA convolution、
gate、recurrent state 或 FFN clamp 的含义与类型。

MTP routed experts 的 GU/down 使用 128×128 块 E4M3FN codes 与 BF16 multiplier：

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

### 2.3 Optimized draft head

从源 BF16 `lm_head.weight[154880,4096]` 直接生成额外的 `q4_g64_fp16` 矩阵，
采用通用 `grouped_absmax`、K64 分组和 FP16 scales，计算许可为 `A16Only`。
保留完整词表及原 token ID 顺序，不经过其他量化表示再次编码。
Main head、embedding 和 proposal head 是独立权重对象；Main/MTP 共享 embedding。

额外 head 作为可选 proposal 绑定，启用对应配置时加载。其输入是 MTP final norm
后的 hidden，不重复执行 norm。词表 TP 分片由并行计划决定，格式不改变输出决策语义。

## 3. MLA／Indexer cache 与计算政策

MLA latent 通过现有 `kv_dtype=bf16|fp8` 选择。Indexer 默认使用 FP8；启动协调进程的
`NINFER_KPOOL_INDEXER_BF16=1` 选择 BF16，未设置或 `0` 使用 FP8。该选择在模型启动时
解析并随原生准备请求传给所有 ranks，不增加 Engine 配置项。Main/MTP 和所有 ranks
使用同一对格式，运行中保持不变；latent 与 index 格式可以独立组合。

| 部分 | 持久表示 | 计算 |
|---|---|---|
| BF16 latent | 原始基底，BF16 `[512]`，1024 B/token | BF16 QK/PV，FP32 累加 |
| FP8 latent | 原始基底，E4M3FN `[512]`＋4 个 UE8M0 scales，516 B/token | 私有 Q_latent 动态 G128 量化，FP8 QK；latent 片上解码为 BF16 供 PV 使用 |
| BF16 pooled index keys | 原始基底，BF16 `[128]`，256 B/pool | BF16 QK，FP32 累加、signed weighting 与 head reduction |
| FP8 pooled index keys | 归一化 H128 基底，E4M3FN `[128]`＋1 个 UE8M0 scale，129 B/pool | Query 使用相同 H128 和 G128 编码，FP8 QK，评分归约 FP32 |
| Raw tail/records | 原始 BF16 K/gate，不含 APE | FP32 pooling；随后按选定 index 格式发布 |

### 3.1 共用编码

所有 FP8 cache 使用 E4M3FN、连续 feature G128 分组和 UE8M0 单字节二次幂 multiplier。
精确编码、零值和极小值规则由 [kv_cache_format.h](../../include/ninfer/kv_cache_format.h)
唯一维护。MLA latent 不旋转；Indexer 的归一化 Hadamard-128 属于其算法，不能推广到
其他 cache。四 token pool 是读取／选择单位，latent 的 scales 不跨 token 共享。

Prepare 直接将 FP32 RMSNorm 结果编码为 latent；FP8 路径不增加中间 BF16 舍入。
Raw K/gate 保留公开 BF16 边界，pooling 消费其实际 words。BF16 pooled key 对加权和
执行 BF16 舍入；FP8 pooled key 将 FP32 加权和旋转后直接编码。

### 3.2 消费与数值边界

Sparse MLA 保留 BF16 U_K/U_V。FP8 路径在 U_K 的 FP32 累加结果上按 G128 编码
Q_latent，FP8 QK 的各组乘积按对应 query/latent scales 还原后以 FP32 归约。
Softmax、split partial 与合并统计为 FP32；PV 使用片上 BF16 latent。
U_V 消费 BF16 latent result，公开输出仍为 BF16 heads。不生成全局展开的 BF16 cache。

Indexer 的公开 query 始终为 BF16，head coefficients 为 FP32。FP8 路径内部旋转和
编码 query；评分的 ReLU、带符号系数、head reduction 保持 FP32。Top512 对实际计算
得到的 scores 精确选择；浮点评分差异与集合选择正确性分别验收。

当前窗口新 token 与历史 token 使用同一 cache 表示，不通过未编码 latent 绕过表示
边界。Codec 检查针对给定 FP32 输入；完整 Prepare、Indexer、MLA 和组合链使用独立
数学参考，不能把私有 query 量化步骤复制进参考来消除误差。

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
pool 出口按第 3 节选定的 index 格式发布，不增加额外 norm。

### 4.3 Latent 域 attention

本模型保留 BF16 U_K/U_V 因子，不离线与 Q_b/O 合成扩大后的吸收矩阵。
计算精度和私有中间值的表示遵循第 3 节；公共输出是完整 U_V 映射后的 BF16 heads，
完整数学见模型第 4.1 节。Attention scale 固定为 1/16，不因 512 维 latent 改成
`1/sqrt(512)`。O projection 消费 BF16 heads，使用 Q8-G32 权重和 A16 计算；
Q_b、U_K/U_V 与 Indexer 投影权重仍为 BF16。
空选择集合的写域和精确零规则由消费者接口规定。

### 4.4 MTP 与 Vision 的融合边界

MTP 的 embedding/conditioning 分别做 FP32 direct-gamma RMSNorm，再输出 BF16
拼接向量；embedding 禁用列的对应分量为正零。Stem 消费这份实际 BF16 输入。
MTP 与 Vision 的普通 residual 为 `r=RN_BF16(FP32(x)+FP32(branch))`；融合的
后续 RMSNorm 消费实际 r，不能使用未舍入的 sum 替代已发布的 BF16 residual。

Vision 的带 bias projection 先在 FP32 accumulator 上加 bias，再写 BF16。
融合 GU 在 FP32 projection/bias 后做 clamp、SiLU 和乘积，最终写 BF16 product，
内部 gate/up 没有额外 BF16 落点。Q/K head RMSNorm 与 2D RoPE 以 FP32 组合，
在旋转出口写 BF16，不为独立 norm 调用添加中间 BF16 舍入。

Merger 的独立 projection、LayerNorm、Exact GELU 及 FFN 输入使用实际 BF16
公共出口。以上是 NInfer 的表示边界，不要求与参考后端的拆分或融合中间值逐位一致。
接口、alias 与模型组合见 [MTP](../designs/glm5next/mtp.md) 和
[Vision](../designs/glm5next/vision.md)。

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
下表是全模型的 tensor payload，单位 GiB；目标列包含额外 Q4 draft head，
不含文件头、对象对齐、runtime 或分片复制。目标列是按已选格式计算的估算，
尚非实际 artifact 大小。

| 部分 | NVIDIA 来源 | 选定表示后的估算 |
|---|---:|---:|
| Main routed experts | 159.469020 | 159.469020 |
| Main dense MLP | 0.237305 | 0.237305 |
| Main KDA | 8.729325 | 4.744950 |
| Main MLA＋Indexer | 2.559384 | 1.914853 |
| Main shared experts | 1.968750 | 1.045898 |
| Main router | 0.092330 | 0.092330 |
| Main mHC＋norm | 0.066617 | 0.066617 |
| Embedding＋LM head | 2.363281 | 2.363281 |
| MTP routed experts | 13.500000 | 6.750824 |
| MTP 其他独有权重 | 0.344283 | 0.263716 |
| Vision | 1.049837 | 1.049837 |
| Optimized draft head | — | 0.313873 |
| **合计** | **190.380132** | **178.312505** |

KDA 的 Q/K/V/O 原载荷恰为 8.5 GiB，Q8-G32 后 4.515625 GiB，减少 3.984375 GiB。
MTP experts 共 7247757312 个权重，block-FP8 codes＋BF16 scales 为
7248642048 B，减少 6.749176 GiB。Main/MTP 共 12 个 MLA O 从 BF16 改为 Q8，
减少 0.703125 GiB；43 个 shared FFN 减少 0.944824 GiB。额外 Q4 head 增加
0.313873 GiB，最终载荷估算为 **191461594820 B**。

不启用 MTP 时应扣除其 experts、其他独有权重和仅供 draft 使用的 proposal head；
未启用 optimized draft head 时不加载该额外 head。Embedding/Main head 不重复扣除。
不启用 Vision 时扣除独立 Vision 权重。双机每 rank 的权重容量还取决于实际分片与复制
计划，不能直接以总量除二作为最终驻留预算。

NVIDIA 模型卡中量化 shared experts 的措辞与实际配置不符，所链接 recipe 中「MTP
未导出」的注释也不符合这份本地 checkpoint。本文采用实际 tensors：源 shared 为
BF16，完整 MTP 权重存在且原 experts 为 BF16。

## 7. 来源与通用合同

源 checkpoint、配置和模型实现入口见[模型来源](glm5next-model.md#8-来源)。
NVFP4 源表示同时参照
[NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)，
实际存在的 tensors 和容量以源文件核对为准。

Q8 的 grouped_absmax、NVFP4 divisor 与已有 block-FP8 represented values 遵循
[通用数值格式](tensor-formats.md)。实际格式已有定义时不另设 GLM 专属解释；尚待
实现的编码规则在代码／通用格式接管后改为直接引用，模型的格式选择长期保留。
模型加载、转换命令和产物状态不由本参考文档宣称，见设计索引及实际 converter。
