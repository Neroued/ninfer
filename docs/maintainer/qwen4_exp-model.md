# Qwen4Exp 模型与权重

本文维护 Qwen4Exp 的模型结构、Flash Next 实例、量化 recipe 和持久权重布局。
当前执行能力、资源所有权与状态恢复见[执行与状态](qwen4_exp-execution.md)；
转换命令见[权重转换](../weight-conversion.md)。通用数值表示、字节布局和文件合同分别由
[数值格式](tensor-formats.md)、[存储布局](storage-layouts.md)、
[V3 容器](artifact-container.md)维护。

## 1. 架构与实例

Qwen3.8-Flash-Next 使用 `Qwen4ExpForCausalLM / qwen4_exp_text`。
Converter 接受源 `Qwen4ExpForConditionalGeneration` 或 `Qwen4ExpForCausalLM`，
统一输出这一架构；Vision 和原生 MTP 是 target 为 `text` 的可选组件。
Qwen3.8-27B 使用另一套 Qwen3.5 架构，两者不因名称相近而共用 Text 层实现。

下表是 Flash Next checkpoint 的实例值。模型执行从 config 读取层序和几何，
Binding 描述实际权重，Parameters/Ops 决定可执行表示与 shape；recipe 名称不参与执行注册。

| 参数 | Flash Next |
|---|---|
| Text hidden / vocabulary | 2560 / 248320；embedding 与 head 不绑权重 |
| Main | 48 层；零基层号 `l%4==3` 为 QSA，共 12 层，其余 36 层为 GDN |
| HC streams / lowrank | 4 / 320；residual 宽度 10240 |
| GDN heads / dimension | Q/K 16 heads、V 48 heads，维度均为 128；每组 Q/K 服务连续三个 V heads |
| GDN QKV / Z / convolution | 10240 / 6144；4 taps、dilation=1 |
| QSA Q / KV | 24 / 2 heads，head dimension=256 |
| QSA indexer | 4 个 query heads、1 个 key head，dimension=128 |
| QSA compression / selection | 四 token 一块；最多 512 个完整块，另加未完成块的至多三个可见 token |
| Routed experts | 每层 512 个，top-10，intermediate=640 |
| Shared expert | 每层一个 SwiGLU，intermediate=640，独立标量 sigmoid gate |
| PLE 位置 | 零基第 1 层的 attention HC 之前；源 `ple_layer_ids=[2]` 规范化为 `[1]` |
| PLE heads / embedding | 2-gram、3-gram 各 8 heads；16 行 × 160 维，拼成 2560 维 |
| PLE table | 128 个 `[2500012,160]` shards，共 320001536 行 |
| PLE convolution | 10240 通道，4 taps、dilation=3，九列历史 |
| MTP | 一个独立 HC/QSA/MoE 层，无 PLE；共享 Main embedding，默认共享 Main head，可选独立量化 draft head |
| Vision blocks | 27 层，hidden=1152，16 heads × 72，MLP intermediate=4304 |
| Vision patch / merger | patch `[3,2,16,16]`；2×2 merge，4608→4608→2560 |
| RoPE | theta=10000000；Main 与 indexer 都旋转前 64 维；interleaved MRoPE sections `[11,11,10]` |
| Context / norm epsilon / EOS | 262144 / `1e-6` / 248044 |

Adapter 将源 `full_attention` 规范化为 `indexed_attention`，并将
`norm_topk_prob=true` 等数学配置写入产物。字段和源参数映射由
[adapter](../../tools/convert/qwen4_exp.py)维护，原生配置解释见
[config](../../src/models/qwen4_exp/config.cpp)。

## 2. 模型计算与数值边界

### 2.1 Main 与 HC

```text
token embedding，在视觉位置替换为 Vision 特征
  → 复制为四条 HC residual streams
  → 每层：attention HC → GDN/QSA → HC combine
          MoE HC       → MoE     → HC combine
          （零基第 1 层入口先加入 PLE）
  → final HC norm/mixer → 2560 维 sample hidden → LM head
```

四条 streams 是特征轴，GDN/QSA/MoE 仍消费一个 2560 维 block 输入。
HC 对每条 stream 独立做 offset RMSNorm；低秩 down/up 产生逐特征的 sigmoid mixing，
block injection 为四个 `2*sigmoid(logit/4)` 系数。Combine 将同一个 block 输出按
这些系数加回各 stream，形成公开的 BF16 residual；下一次 norm 消费这些实际 BF16 words。
Final mixer 不产生 injection，其后没有额外 final RMSNorm。

[HC 合同](../../include/ninfer/ops/gated_residual.h)维护完整公式。
模型保留两种末端特征：final mixer 的 2560 维输出，以及其 norm/mixer 之前的
10240 维 residual。MTP conditioning 使用后者，不能从前者反推。

### 2.2 GDN、QSA 与 MoE

GDN 将 HC 输入投影为 QKVZ 和 AB。Q/K/V 经过因果卷积与 SiLU，再做 Q/K L2
normalization 和 gated delta-rule；Z 绕过卷积。Recurrent state 为
FP32 `S[value_dim,key_dim]`。输出使用直接 gamma 的 head-wise RMSNorm，
乘 `sigmoid(Z)` 后投影回 2560 维。卷积 SiLU 与输出 sigmoid 是两个不同的非线性。
计算合同见 [GDN](../../include/ninfer/ops/gated_delta_net.h)、
[controls](../../include/ninfer/ops/gdn_control.h)及
[gated RMSNorm](../../include/ninfer/ops/rmsnorm.h)。

QSA 保留完整主 K/V 历史；4:1 压缩仅作用于 index key。每个完整块先将四列 raw index-K
平均并舍入为 BF16，再做 offset norm 和块首坐标 RoPE。四个 index heads 的非负 dot
之和决定共享 TopK；所有 24 个 query heads 消费同一选块集合。Sparse GQA 还读取当前
未完成块的可见 token，输出乘逐元素 sigmoid gate 后进入 O projection。
Index score 不加入主 attention logits。完整公式与 KV codec 边界见
[QSA 合同](../../include/ninfer/ops/qsa.h)。

MoE 使用归一化 top-k softmax 路由，shared 不参加 top-k：

```text
expert_e(x) = W_down_e * (SiLU(W_gate_e*x) * (W_up_e*x))
routed(x)   = sum_selected_e(prob_e * expert_e(x))
shared(x)   = sigmoid(W_shared_score*x) * shared_ffn(x)
moe(x)      = routed(x) + shared(x)
```

Shared FFN 同样使用 SwiGLU。MoE 输出一个 2560 维 block，由 HC 回注 residual。
投影、非线性、route coefficient 和逐 expert 归约的私有 staging 不增加公共 BF16
舍入边界；语义入口及 activation policy 由 [MoE 合同](../../include/ninfer/ops/moe.h)维护。

### 2.3 MTP 与 Vision

MTP 的初始特征为：

```text
e = W_fc_embedding * OffsetNorm_2560(token_embedding)
v = OffsetNorm_10240(multistream_hidden).reshape(4,2560)
h_init[s] = W_fc_hidden * v[s] + e
```

Hidden norm 沿完整 10240 维归约，随后同一个 fc_hidden 独立作用于四条 streams。
第一步使用 Main conditioning，后续 draft step 使用前一步 MTP 的 pre-final-mixer
residual。MTP 的 QSA/MoE 层及 final HC mixer 产生下一组 conditioning 和 sample。
Draft 默认使用 Main head，也可选择第 3.3 节的独立量化 head；target 仍使用 Main head。

Vision 将 Conv3d patch 按 C/T/H/W 展平为 1536 维。Blocks 使用非因果 attention、
LayerNorm 和 `gelu_pytorch_tanh` MLP；merger 在 2×2 合并前做 LayerNorm，
两次 FC 之间使用 GELU。本实例没有 deepstack。编码器复用
[公共 Qwen Vision](../../src/models/common/qwen/vision/)，Text 层只消费其 2560 维输出。

## 3. Flash Next 量化 recipe

`qwen3_8_flash_next` 使用 BF16 主 checkpoint 提供 config、资源、Main 非 routed
权重、Vision 和 PLE；`--source nvidia=PATH` 提供完整
`nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint，导入 Main routed experts 和 MTP 独有参数。
来源和转换方法写入 `.conversion.json`。完整选择由
[official recipe](../../tools/convert/official_recipes.py)维护。

| 部分 | 持久表示 | 计算许可或消费 |
|---|---|---|
| Main routed gate/up/down | NVIDIA NVFP4，G16 E4M3FN scale，每 expert 矩阵一个 FP32 divisor | `AllowA4` |
| Main GDN QKVZ/output、QSA Q/gate/output | BF16 源转 Q8 G32/FP16 scale | `A16Only` |
| HC、QSA K/V/indexer、PLE 投影、router/shared FFN | 原 BF16 words | `A16Only` |
| GDN AB、convolution、norm，其他 norm/bias | 原 BF16 words | 各 Op 数学 |
| GDN A_log / dt_bias | 源 BF16 精确展开为 FP32，各 `[48]` | 保留原始含义 |
| Token embedding / Main LM head | 两份独立 BF16 对象 | lookup / A16 head；MTP 默认共享 |
| Optimized MTP head，可选 | BF16 Main head 源转 Q4 G64/FP16 scale，完整词表 | `A16Only` |
| PLE table | BF16 源转 `nf4_g16_e4m3fn` | 精确解码为 BF16 lookup 输出 |
| MTP routed experts | NVIDIA E4M3FN codes，128×128 块 BF16 multiplier | `AllowA8` |
| MTP 其他独有参数、全部 Vision | 原 BF16 words | 投影 `A16Only` |
| PLE hash controls | 原始 35 个 INT64 | 无损 Resource |

GPU MoE 已有 NVFP4 A4、block-FP8 A8/G128 及小批量 A16 路径；
各语义入口的私有 shape policy 选择实际计算。CPU Main experts 使用 BF16 输入和
FP32 中间计算，不量化激活。许可允许高精度路径，并不要求每次调用都量化为最低位宽。

Q8 使用 `grouped_absmax`，沿 K 每 32 个值分组。合并和行置换先确定最终行序，
再进行编码。Packing、scale swizzle 和 convolution 转置在转换时完成，loader 绑定原字节。
KV、recurrent state 和 workspace 的 dtype 独立于权重 recipe。

### 3.1 Norm 参数

| 角色 | 持久参数 | 归约与 affine |
|---|---|---|
| HC、PLE grouped RMSNorm | BF16 offset `[10240]`，native `[2560,4]` | 各 stream 独立归约，gain=`1+w` |
| Main/MTP QSA Q/K、indexer Q/K | BF16 offset `[256]` / `[128]` | 每 head 归约，同角色 heads 共享参数 |
| MTP pre-FC embedding / hidden | BF16 offset `[2560]` / `[10240]` | 对整个对应向量归约 |
| GDN output gated RMSNorm | BF16 gamma `[128]` | 直接 gain，所有 V heads 共享 |
| Vision LayerNorm | BF16 weight/bias `[1152]` | 直接 affine；merger norm 位于 merge 前 |

Offset 保留源 words，`1+w` 不预先舍入为 BF16 gain；HC 的 `/4` 不折入权重。
参数 dtype 与归约精度是不同的约定。

### 3.2 NVIDIA 编码导入

Main 源的 `weight_scale_2`、`input_scale` 是正有限 FP32 multipliers。
[ModelOpt adapter](../../tools/convert/sources/modelopt_nvfp4.py)使用：

```text
source weight = E2M1(code) * E4M3FN(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
stored weight = E2M1(code) * E4M3FN(scale) / d_w
```

Codes/scales words 保留后按目标 layout 重排；FP32 reciprocal 舍入是元数据变换，
不承诺所有解码值与源逐位相同。Recipe 要求同 expert 的 gate/up 两个 multiplier
words 相等，并要求一层所有 experts 的 GU 输入 multiplier 相等、down 输入 multiplier
相等。每个 expert 仍各自持有 weight divisor。

Activation divisors 存为 FP32 `[48,2]`，元素 `2*l` 与 `2*l+1` 分别供 GU/down Use
引用；它们与 weight divisor 分开解释。A4 输入按 G16 量化，product 从 FP32 直接编码，
不增加 BF16 中间 seam。

MTP `fp8_e4m3fn_block128x128_bf16` 重建为 E4M3FN code × BF16 block scale。
Gate/up 各自保留原 scale，640 行边界与 128 行块对齐。A8 按每 token 的 K128 分组
动态量化，scale 为 FP32；每个 K128 dot 分别缩放后累加，没有静态 activation divisor。

### 3.3 Optimized MTP head

转换时 `--proposal` 从 Main BF16 head 源生成独立 `proposal/head`，使用
`grouped_absmax`、`q4_g64_fp16` 和 `row_split_k128_v1`。它保留全部
`[248320,2560]` 物理行及原 vocabulary ID 顺序，不裁剪词表、不创建 `proposal/token_ids`；
目录声明 `text.proposal.domain="full"`。`mtp/final_hidden` Use 的 policy 为 `A16Only`。

Main BF16 head 和 `mtp/output_head` 的共享绑定保留。启动选择 optimized 时，
MTP 改为绑定 `proposal/head`，target 验收和 ordinary decode 继续使用 Main head。
两份 head 采用相同的词表 TP 分区：TP2 下每 rank 各持有 `[124160,2560]` 的本地部分，
沿用全局 token ID 和候选汇合，不需要 token remap。

这是附加的 draft 表示；关闭 optimized head 时不加载其对象。
启动选项和执行流程见[执行与状态](qwen4_exp-execution.md#5-mtp-的生成与采用)。

## 4. 合并矩阵与持久布局

逻辑矩阵写作 `W[N,K]`；N 是输出行，K 是输入特征，区间均为半开区间。
相同输入、表示和兼容元数据的投影沿 N 合并，逻辑角色仍由各自 Binding 引用。
Native Tensor 第 0 维连续；`M=T*B`，`column=t+T*b`。

| Parent | Main / MTP 数量 | 逻辑 shape | 格式 | 输出行顺序 |
|---|---:|---|---|---|
| GDN QKVZ | 36 / 0 | `[16384,2560]` | Q8 | Q 2048、K 2048、V 6144、Z 6144 |
| GDN AB | 36 / 0 | `[96,2560]` | BF16 | a 48、b 48 |
| GDN output | 36 / 0 | `[2560,6144]` | Q8 | 原行序 |
| Main QSA Q/gate | 12 / 0 | `[12288,2560]` | Q8 | Q 6144、gate 6144 |
| Main QSA KV/index | 12 / 0 | `[1664,2560]` | BF16 | K 512、V 512、index-Q 512、index-K 128 |
| Main QSA output | 12 / 0 | `[2560,6144]` | Q8 | 原行序 |
| MTP QSA input | 0 / 1 | `[13952,2560]` | BF16 | Q 6144、K 512、gate 6144、V 512、index-Q 512、index-K 128 |
| MTP QSA output | 0 / 1 | `[2560,6144]` | BF16 | 原行序 |
| Block HC down/inject | 96 / 2 | `[324,10240]` | BF16 | down 320、injection 4 |
| Final HC down | 1 / 1 | `[320,10240]` | BF16 | 全部 down |
| HC up | 97 / 3 | `[10240,320]` | BF16 | stream-major，`row=s*2560+d` |
| PLE KV | 1 / 0 | `[12800,2560]` | BF16 | 四流 key 10240、共享 value 2560 |
| MoE input | 48 / 1 | `[1793,2560]` | BF16 | router 512、shared score 1、shared gate 640、shared up 640 |
| Shared down | 48 / 1 | `[2560,640]` | BF16 | 原行序 |
| Routed GU bank | 48 / 1 | `[512,1280,2560]` | Main NVFP4 / MTP block-FP8 | expert-major，每 expert 内 gate 640、up 640 |
| Routed down bank | 48 / 1 | `[512,2560,640]` | Main NVFP4 / MTP block-FP8 | expert-major |
| MTP fc_embedding / fc_hidden | 0 / 各 1 | `[2560,2560]` | BF16 | 独立权重；分别处理 M / 4M 列 |
| Main LM head | 一份，MTP 默认共享 | `[248320,2560]` | BF16 | vocabulary 顺序 |
| Optimized MTP head | 0 / 可选 1 | `[248320,2560]` | Q4 | 完整 vocabulary 顺序 |

Main QSA 的两个 input parents 精度不同；MTP 六段投影合为一个 BF16 parent。
MoE control 为前 513 行，shared GU 为后 1280 行，可作零拷贝 views。
持久合并不强制一次投影调用；LinearSplit 可以直接产出独立连续的激活分段。

### 4.1 行置换与 convolution

源 q_proj 按 head 交错 Q/gate，转换时解交错：

```text
Q[h*256+d,:]    = source_q_proj[h*512+d,:]
Gate[h*256+d,:] = source_q_proj[h*512+256+d,:]
```

GDN 与 PLE 的 convolution 都转为 channel 连续：

```text
source [C,1,4] → artifact/Binding [4,C] → native Tensor [C,4]
stored[j*C+c] = source[c*4+j], C=10240
```

这两类变换均保留源 BF16 words；卷积 tap 时间顺序不变。PLE 的 dilation=3 由计算解释。
Norm、bias 与 convolution 不作为 Linear 权重重新量化。

### 4.2 Vision 与 expert banks

| Vision 参数 | 数量 | 持久 shape |
|---|---:|---|
| Patch projection | 1 | `[1152,1536]` |
| QKV / bias | 27 | `[3456,1152]` / `[3456]`，Q/K/V 依次排列 |
| Attention output | 27 | `[1152,1152]` |
| FC1 / FC2 | 各 27 | `[4304,1152]` / `[1152,4304]` |
| Merger FC1 / FC2 | 各 1 | `[4608,4608]` / `[2560,4608]` |
| Position embedding | 1 | `[2304,1152]` |

每层 routed experts 有 GU、down 两个物理 parent；逻辑 gate/up 各
`[512,640,2560]`，down 为 `[512,2560,640]`。Binding 按 expert ID 引用 GU 的两半
和 down 全矩阵。Binding range 按 C-order 逻辑元素计数，不包含 scale/divisor bytes。

Main 使用 `batched_block_scale_k16_m128x4_v1`，bank 内依次存所有 codes、
swizzled scales、FP32 divisors `[512]`。GU/down payload 分别为
943720448 / 471861248 B。Scale swizzle 使用 expert 内的局部行号。
MTP 使用 `block_scale_m128_k128_v1`，codes 后为 BF16 scales
`[512,10,20]` / `[512,20,5]`，payload 分别为 1677926400 / 838963200 B。

Recipe 按资源/辅助值、Main 固定权重、Vision、MTP、Main experts、PLE shards 放置。
Main experts 按 layer→GU/down，PLE 按 shard ID；普通对象对齐 256 B，PLE 对齐 4096 B。
对象可以跨 V3 续卷，实际文件范围由目录映射，不能从文件名推导参数身份。

## 5. PLE 表、hash 与 lookup 出口

PLE 的 128 个 NF4 parents 保留原 shard 边界和各自 divisor。
`stored_scale_mse` 对每 shard 做有界两遍转换；误差计算、scale 搜索和平局规则见
[转换数值合同](tensor-formats.md#36-stored_scale_mse-conversion-method)。

`row_paged_g16_4k_v1` 每行 90 B：80 B packed codes、10 B scales。
每个 4 KiB 数据页存 45 行，末页 37 行；每 shard payload 为 227557380 B。
全局行 g 属于 `shard=g/2500012`、`row=g%2500012`，hash head 不对应固定 shard。
具体 padding、divisor 地址和编码见[布局合同](storage-layouts.md#6-row_paged_g16_4k_v1)。

`text.resources.ple_hash_controls` 是 280 B 的 `raw_bytes_v1` Resource，顺序为：
三个 layer multipliers、十六个 head vocab sizes、十六个 head offsets，
均为源 little-endian signed INT64 words。控制值无损复制，不重新生成 seed/prime。

当前位置 raw ID 为 x0，前一、前二枚为 x1、x2；负位置补 EOS。
若 x1=EOS，仅对本次查询令 x2=EOS；x0=EOS 仍消费此前历史：

```text
u2 = word64(x0*multiplier[0]) XOR word64(x1*multiplier[1])
u3 = u2 XOR word64(x2*multiplier[2])
g[h] = positive_mod(signed64(h<8 ? u2 : u3), head_vocab_size[h])
       + head_offset[h]
```

乘法保留 modulo-2^64 words，再解释为 signed64 做正模。输出按 2-gram 八个 heads、
3-gram 八个 heads 拼接。分段与增量 append 使用真实 raw-ID tail；视觉特征不替换 IDs，
EOS 规则不清空 PLE convolution history。实现见 [hash](../../src/models/qwen4_exp/ple/hash.cpp)。

Lookup 出口严格为：

```text
LUT[shard,scale_word,code] =
    BF16_RNE(codebook[code] * E4M3FN(scale_word) / divisor[shard])
```

它对精确表示值直接做 BF16 RNE，不能用未经证明的 FP32→BF16 双舍入替代。
Scale words 的有效域为 `0x00..0x7e`；零结果保留 codebook 的符号。
每 shard 的不可变 LUT 占 4096 B，合计 512 KiB；行解码写出 160 个 BF16 words，
十六行按 head 顺序组成 `E[2560,T,B]`。通用实现见
[NF4 Host 数值辅助](../../src/core/nf4.h)，异步准备和寿命见执行文档。

## 6. 权重容量与驻留

下面是此 recipe 的物理权重及控制值，不包含 frontend resources、对象间对齐、
目录和文件 framing。它不是启动时的总内存占用。

| 部分 | Bytes | GiB | 当前放置 |
|---|---:|---:|---|
| Main 非 routed-expert、非 PLE 表权重 | 7418143232 | 6.908684 | GPU |
| Main routed experts | 67947921408 | 63.281433 | GPU resident，或 registered Host backing 加 RankProgram GPU cache |
| PLE NF4 表 | 29127344640 | 27.126954 | V3 磁盘原位 backing |
| Vision | 897862112 | 0.836199 | 启用时 GPU |
| MTP experts | 2516889600 | 2.344036 | 启用时 GPU |
| MTP 其他独有权重 | 181136896 | 0.168697 | 启用时 GPU |
| Activation divisors 与 PLE hash controls | 664 | — | 只读控制值 |
| 不含 optimized head 合计 | **108089298552** | **100.666004** | — |
| Optimized MTP head，可选 | 337715200 | 0.314522 | 启用时 GPU；TP2 每 rank 一半 |
| 含 optimized head 合计 | **108427013752** | **100.980526** | — |

Text-only 权重及控制值约 97.317072 GiB。全部组件在 offload 模式的固定 GPU 权重约
10.257616 GiB，另计 Main expert cache；resident 模式的 GPU 权重约 73.539049 GiB。
两种模式均另计 StateImage、KV、workspace、Graph 和 staging。
未启用的 Vision/MTP 不 materialize；共享 embedding/head 不重复计费。
启用 optimized head 在上述 GPU 权重之外增加约 322.07 MiB；
TP2 每 rank 增加约 161.04 MiB。

Offload 模式的 Host backing 包含已缓存到 GPU 的 experts，eviction 无需写回；resident
模式直接绑定 GPU bank，不保留 Host expert 副本。PLE 启动只读目录、控制值和 divisors，不整体加载表。
ModelDefinition 持有配置、表示描述与 Host 控制值；ModelShard 持有本地不可变权重与 PLE 文件 backing。
Session 管理逻辑状态和上下文目录，RankProgram 持有可变缓存和物理执行资源。
完整资源预算见[执行与状态](qwen4_exp-execution.md)。

## 7. 数学来源

源结构参照固定版本的
[Transformers 模型](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)、
[配置](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)及
[vLLM MTP](https://github.com/vllm-project/vllm/blob/6fddc9b21e763bfc947cc01d3ce815179aac1dc6/vllm/models/qwen4_exp/nvidia/mtp.py)。
[NVIDIA 来源说明](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/README.md)
记录量化来源。NInfer 的正式数值边界由生产 Op 合同维护；上游 TP、私有 casts 和
kernel packing 不自动成为本项目的数学或持久合同。
