# GLM5Next Vision 输入、计算与并行设计

本文确定 GLM Vision 的媒体准备、encoder/merger 计算、Op 扩展、双机任务划分与
资源交接。它是待实现设计；完整模型数学见[模型参考](../../maintainer/glm5next-model.md#62-vision)，
表示与舍入见[数值参考](../../maintainer/glm5next-quantization.md)，权重布局与
放置由[并行设计](parallel.md)维护。

## 1. 几何与计算边界

| 项目 | 当前 checkpoint |
|---|---|
| 输入 | RGB，temporal patch=2，spatial patch=14×14，每 patch 1176 features |
| Encoder | 24 blocks，hidden=1024，16 heads，head_dim=64 |
| Block FFN | intermediate=4096，gate/up clamp limit=10 |
| Spatial merge | 2×2，四个 1024 维 patch 合为一个 4096 维向量 |
| Merger | 4096→4096 projection；LayerNorm＋Exact GELU；4096→10240→4096 FFN |
| RMSNorm | 直接 gamma；block、Q/K head 和 post norm 的 epsilon=1e-5 |
| Merger LayerNorm | affine weight＋bias，epsilon=1e-5 |
| RoPE | 2D split-half，theta=10000，覆盖完整 D64 |
| 权重与激活 | BF16，projection A16，norm/softmax/归约为 FP32 |

参数由 config、processor 资源和权重绑定提供，不在 Engine 中再次声明支持表。
当前 Vision 权重全部保留 BF16，包括 bias 和 norm。

Vision 是模型执行模块，调用 Linear、Normalization、RoPE 和 SoftmaxAttention。
它没有生成期 KV、KDA state 或 StateImage 字段。Encoder 的非因果 attention 与
文本 Kpool sparse MLA 的状态及数学不同，使用现有 dense attention family。

## 2. 从媒体到文本 embedding

完整流程：

```text
encoded image/video
  → 通用媒体解码 + GLM 采样/resize-pad/normalize/patchify
  → BF16 patches [1176,N]
  → Patch LinearBias [1024,1176]
  → 24 个 Vision blocks
  → post RMSNorm
  → 连续 view [1024,N] → [4096,N/4]
  → Downsample LinearBias [4096,4096]
  → Merger Linear [4096,4096]
  → affine LayerNorm → Exact GELU
  → clamped GU [20480,4096] → down [4096,10240]
  → features [4096,N/4]
  → 按媒体 spans 替换 Main token embedding
  → Main mHC expand 与后续文本层
```

N 是 raw patch 数，不是合并后的媒体 token 数。各阶段使用同一份预处理计划产生的
grid、patch 次序、attention segments 和 token spans；预算与执行不各算一套几何。

### 2.1 一个 encoder block

用 x 表示当前 BF16 [1024,N]：

```text
n1       = RMSNorm(x, gamma1)
(Q,K,V)  = split(Wqkv*n1 + bqkv)
Q        = RoPE2D(RMSNorm_head(Q, gamma_q))
K        = RoPE2D(RMSNorm_head(K, gamma_k))
A        = SegmentAttention(Q,K,V, scale=1/8)
branch_a = Wo*A + bo
r        = BF16(x + branch_a)

n2       = RMSNorm(r, gamma2)
g,u      = Wgu*n2 + bgu
z        = SiLU(min(g,10)) * clamp(u,-10,10)
branch_f = Wdown*z + bdown
xnext    = BF16(r + branch_f)
```

Q/K/V 的公共出口分别连续 BF16 [64,16,N]。RMSNorm_head 按每列、每个 head 的
64 项归约，gamma_q 与 gamma_k 各为 [64]，所有 heads 共享对应 gamma。

Attention 是独立 segments 内的完整非因果 attention。O 和 FFN down 的 bias 在
该 projection 出口前加入；GU bias 在 clamp 和非线性前加入。
Gate 只有上界 clamp，不能误写为 gate/up 都做对称 clamp。

第一层的 norm1 独立执行。之后可将 residual add 与接下来所需 RMSNorm 合为
[residual_add_rmsnorm](mtp.md#4-residual-add-与-rmsnorm)：
attention 分支后使用本层 gamma2，FFN 分支后使用下一层 gamma1；
最后一层使用 post norm gamma。模型决定 gamma 和层序，Op 只执行显式公式。

### 2.2 Downsample 与 merger

Encoder 在全部 blocks 后先做 1024 维 post RMSNorm，再把连续四个 spatial patches
解释为 [4096,N/4]。Downsample 使用带 bias 的 [4096,4096] 矩阵。

Merger 顺序固定为：

```text
a = Wproj * downsample_out
b = LayerNorm(a, gamma, bias, eps=1e-5)
c = ExactGELU(b)
z = SiLU(min(Wgate*c,10)) * clamp(Wup*c,-10,10)
features = Wdown*z
```

Merger 的 projection、GU、down 没有 bias，LayerNorm 有 bias；没有额外 residual。
Exact GELU 为 `0.5*x*(1+erf(x/sqrt(2)))`，不能默认使用 tanh 近似。
首版复用独立 LayerNorm 和 Exact GELU，其公共 BF16 边界保持明确。

## 3. Processor 与输入布局

### 3.1 参数与 resize/pad

Frontend 读取 artifact 封装的 processor 配置。当前默认：

| 参数 | 值／行为 |
|---|---|
| 空间 patch／merge | 14／2，patch_expand_factor=1 |
| Canvas 对齐 | 每个空间轴向上对齐到 28 |
| Resize mode | pad；BICUBIC 缩放内容并保持宽高比，右侧与下侧补零 |
| Image token 范围 | 16–8000 个合并后 tokens |
| Video 来源配置 | 最大 240000 tokens，fps=2 |
| Rescale | 1/255 |
| RGB mean | [0.48145466, 0.4578275, 0.40821073] |
| RGB std | [0.26862954, 0.26130258, 0.27577711] |

Pixel-volume 预算与 token 预算的换算因子为
`patch_size² * merge_size² * temporal_patch_size`，使用采样/补帧后的时间尺寸。
普通图像复制为两个相同帧，因此 Tg=1。

先决定对齐 canvas，再在 canvas 内按比例放置内容。超过预算时选择满足预算的
对齐尺寸；低于最小预算时按 processor 规则放大。常规 pad 不为填满 canvas 强行
放大内容。补零发生在 rescale/normalize 之前，归一化后的 padding 通常不为零。

移植 GLM smart_resize 的向上对齐、预算内缩放和 pad 规则，并使用 processor 指定的
resampling 行为。现有 Qwen 的 patch=16、resize 和位置规则不能作为 GLM 默认值。
本地 vLLM 的 30000-video-token 限制是其资源策略，不写入 GLM 模型语义；
NInfer 的请求预算由 Frontend/媒体资源合同执行。

### 3.2 视频采样与 timestamp

GLM processor 拥有 frame 选择，通用 decoder 只负责解码被选中的 frames。
按来源的 `glm_sample_frame_indices` 规则：

1. 根据 duration、target fps 和 frame 上限形成目标数量。
2. 在 source frame 时间轴按 GLM 间隔选择；数量偏离目标时按规定均匀重采样。
3. 短片源 frame 数不足时使用按比例 floor 的选择，再按顺序去重。
4. 选择数量为奇数时重复末帧，补齐 temporal patch=2。

这里的均匀选择分支与 GLM 规则保持一致，不用 Qwen 当前 sampler 替代。
有效但极短的非空视频至少形成一个 temporal patch；目标数量取零时选首帧并复制，
空视频由 decoder 拒绝。此边界不沿用参考实现产生空 selection 的行为。

记录实际 source indices 及对应 timestamps；补出的末帧沿用原 frame 的 timestamp。
按本地 GLM prompt 展开规则，grid g 取其第一帧的整秒值
`floor(source_indices[2*g]/source_fps)`，格式化为一位小数的 `seconds` 文本。
使用已经确定的 indices，不为构造时间文本再运行另一套 sampler；grid 数量与
features 顺序由同一计划派生。

### 3.3 Patch 的 K 轴和列顺序

单个 patch 的 feature 按以下顺序连续：

```text
(channel, temporal, patch_y, patch_x)
k = (((channel*2 + dt)*14 + py)*14 + px)
```

按 FP32 完成 rescale/normalize，转换成实际 BF16 patch words 后交给 projection。
Patch 列的遍历次序：

```text
temporal_grid
  → block_y
  → block_x
  → dy in [0,2)
  → dx in [0,2)
```

其中 `patch_y_index=2*block_y+dy`，
`patch_x_index=2*block_x+dx`。同一 2×2 merge group 的四个 patches 连续，
组内 `p=2*dy+dx`。Processor 直接输出 feature 连续的 BF16 [1176,N]。

Patch projection 的 [1024,1176] 权重保留源 (channel,temporal,y,x) 顺序。
Encoder 保持列顺序；post norm 后 [1024,N] 到 [4096,N/4] 是同一 backing 的 view，
无需 CUDA kernel 或全量中间 Tensor。Downsample 权重的保值重排由
[parallel §2.1](parallel.md#21-vision-权重的保值布局)定义，K 轴恰好消费此 view。

### 3.4 Attention segments 与 2D positions

一个 item 的 grid 为 [Tg,Hg,Wg]：

Tg 为正，Hg/Wg 是正的偶数，由 canvas 对齐和 patch 尺寸共同保证。

```text
raw patches    = Tg * Hg * Wg
merged features = Tg * Hg * Wg / 4
attention segments = Tg
每段长度       = Hg * Wg
```

每个 temporal grid 独立 attention，不跨 grid、图像或视频。Conv3d temporal=2
已经在单 patch 内合并两个 frames，不额外进行跨 temporal-grid attention。

Device `cu_seqlens` 为 I32 [S+1]，每段严格递增，末项为本次执行的 N。
位置为 I32 [N,2]，按上述列顺序存储 y 和 x 两条连续轴，沿时间重复空间坐标。
这里遵循 Tensor 的 feature-first 约定：`positions.ne={N,2}`，
物理地址为 `positions[t + axis*N]`。不将交错 [y0,x0,y1,x1,...] 直接绑定为该 view。

位置值对应 patch grid，而不是 merge 后 grid。没有 learned absolute position，
也没有时间轴 RoPE。Main 文本使用普通 token ordinal，不构造 Qwen MRoPE 三轴位置。

## 4. RoPE 的精确数学与接口

D=64，split-half 配对为 `(i,i+32)`，i=0..31。
令 `j=i%16`：

```text
axis(i) = y, i<16; x, i>=16
phi_i   = axis(i) * theta^(-2*j/32)
out[i]    = n[i]*cos(phi_i) - n[i+32]*sin(phi_i)
out[i+32] = n[i+32]*cos(phi_i) + n[i]*sin(phi_i)
```

前 16 对使用 height，后 16 对使用 width。来源构造器的 partial_rotary_factor=0.5
用于建立单轴频率，再组合两个轴；最终旋转完整 64 维，不是只转前 32 维。

新增 norm+RoPE 入口属于现有 RoPE family：

```cpp
void rmsnorm_rope_2d(
    const Tensor& positions,
    const Tensor& q_weight, const Tensor& k_weight,
    float epsilon, float theta,
    Tensor& q, Tensor& k, CudaExecutionView execution);
```

q/k 连续 BF16 [D,H,N]，本轮支持 D=64、H=16、N≥0；gamma 为 BF16 [64]，
positions 按第 3.4 节布局，epsilon/theta 正且有限。对每个 head 先以 FP32 做
direct-gamma RMSNorm，再进行上述 FP32 旋转，最终写 BF16；中间 n 没有额外 BF16 落点。

q/k 完全原位覆盖，彼此以及与只读输入 disjoint；两个只读 gamma 可以共享 backing。
q/k 基址至少 4 B 对齐，weights 自然对齐，positions 按 I32 对齐；所有对象同设备。
无 workspace、持久状态或隐藏分配；N=0 无设备工作。

同时为现有纯 `rope` 的 Vision2D 语义增加 D64，保留相同坐标/角度约定，
供独立调用和组合对照使用。当前 text `rmsnorm_rope` 的 D128/1D profile
保持原合同；新公式不通过修改它的常量暗中替换。

## 5. Linear 的 bias 与 activation 接口

纯 Linear 几何先按 [Op 清单](op-checklist.md)的 V1–V8 实现/调优；
以下是在相同数学矩阵上的扩展。各 Op 保留自己的分派和调优选择，
共享 contraction/epilogue 计算体，不共享私有调优记录。

### 5.1 LinearBias

```cpp
void linear_bias(
    const Tensor& x, const Weight& weight, const Tensor& bias,
    Tensor& out, CudaExecutionView execution);
```

`Y[n,t]=RN_BF16(sum_k W[n,k]*X[k,t]+bias[n])`。
x/out 连续 BF16 [K,M]/[N,M]，bias 连续 BF16 [N]；
本轮权重为 BF16、Contiguous、A16Only，bias 在 FP32 accumulator 上加入，
然后才输出 BF16。M>0，矩阵基址 16 B 对齐，bias 自然对齐。

使用点为 patch projection、attention O、block FFN down、downsample。
不通过现有无 bias Linear 加上独立 add_bias 建立生产数值边界。

### 5.2 带 bias 的 LinearSplit

在现有 LinearSplit 增加显式 bias 形式：

```cpp
void linear_split(
    const Tensor& x, const Weight& weight, const Tensor& bias,
    std::span<const Tensor> outputs, CudaExecutionView execution);
```

输出是 parent 连续行段，bias [N] 与 parent 行序相同。
每段在 FP32 累加中加入对应 bias，再输出 BF16；段行数为正且和等于 N。
本轮支持单 BF16 parent [3072,1024]，输出 rows=[1024,1024,1024]，
分别可 view 为连续 [64,16,Npatch]。不物化 packed parent 再拆分/复制。

输入、权重及 bias 保持不变，所有 outputs 彼此 disjoint 并与输入 disjoint。
输出大小和布局遵循现有 LinearSplit 的连续矩阵合同。

### 5.3 带 bias 的 clamped LinearSwiGLU

在既有 clamped 入口增加 bias 参数：

```cpp
void linear_swiglu(
    const Tensor& x, const Weight& gate_up_weight, const Tensor& bias,
    Tensor& out, float limit, CudaExecutionView execution);
```

Weight [2I,K]、bias [2I] 的顺序均为 [gate;up]，out 为 BF16 [I,M]。
本轮使用 BF16/A16，limit 正且有限：

```text
g = FP32_GEMM(W_gate,x) + FP32(b_gate)
u = FP32_GEMM(W_up,x)   + FP32(b_up)
out = RN_BF16(SiLU(min(g,limit)) * clamp(u,-limit,limit))
```

GU 是内部值，不强制先存 BF16。当前 block 使用 I=4096、K=1024、limit=10。
Merger 的 I=10240、K=4096 使用既有无 bias clamped 入口，增加对应几何即可。

所有输出与只读输入 disjoint。这三个 BF16/A16 bias 接口无需 workspace，因此不传入
arena、不提供容量查询；执行不隐藏设备分配或权重重打包，支持 CUDA Graph replay。
现有无 bias 接口和数值语义保留，不新增“是否有 bias”的模型侧分派表。

## 6. Dense attention 与既有 Op 复用

现有 SoftmaxAttention 的三个入口扩展为 D64/Hq=Hkv=16、scale=1/8：

- 单 segment 的 `softmax_attention`。
- Device cu_seqlens 的 `packed_softmax_attention`。
- 等长 segments 的 `packed_softmax_attention(...,segment_length,...)`。

q/k/v/out 的 BF16、stride、alias 和 softmax 数值合同沿用现有接口，只有注册几何
增加。不同媒体尺寸使用 packed；相同 grid 可使用等长形式减少 descriptor 准备。
容量查询必须同时覆盖新几何，不能只改执行入口。一般 N 按 Tensor/资源范围支持，
4096 是调优点，不是媒体 patch 上限。

其余计算复用 `rmsnorm`、`layer_norm`、`gelu(GeluMode::Exact)`、普通 Linear、
clamped LinearSwiGLU 和 embedding substitution。Residual/norm 使用 MTP 文档定义的
共享入口。Norm、RoPE、bias 和 attention 的组合无请求状态，支持 CUDA stream/Graph；
本阶段 Vision 的模型调度按 eager 组织。

## 7. 共享媒体基础与 Frontend 所有权

### 7.1 Prepared media 的共同数据

从 Qwen 专有目录抽取真正共享的 owning payload、grid、token span 与媒体身份，
放入 `models/common/frontend/` 的媒体数据文件。GLM 和 Qwen 分别拥有 processor、
tokenizer/template、位置解释和预算参数。

当前 `PreparedMediaPayload` 的 BF16 backing 可直接保留，新增明确的
`features_per_patch`；`patch_elements/features_per_patch` 得到列数。
它不是固定 1536 features 的 Qwen payload，也不需要为本阶段加入任意 dtype 系统。

共同数据包含：

| 数据 | 权威来源 |
|---|---|
| Owning BF16 patches、feature 宽度、元素数 | Processor materialization |
| Item grid、modality、source content digest | 同一 prepared item |
| 视频 source indices/timestamps | GLM frame-selection 计划 |
| Token spans、expanded token 数及媒体位置 | GLM Frontend 的模板展开 |
| 执行 segments、patch/feature ranges | 从 grid 和 spans 派生的模型 Vision plan |

Prepared payload 不携带 CUDA 地址、rank 或 stream。任务和缓存持有不可变共享
payload，生命周期覆盖异步上传及其最后消费者。Cache identity 包含内容身份和影响
结果的 processor 参数、sampling/grid 等；仅比较媒体占位 token IDs 不足以复用内容。

删除公共使用点对 Qwen patch 常量的依赖；Qwen 自己的数值、预算和 position 规则
继续由 Qwen processor 提供。共享不要求统一两个模型的 resize 或模板语言。

### 7.2 通用 decoder 接口

现有 decoder 在 `decode_video(target_fps,min_frames,max_frames)` 中选择 frames。
改为 processor 选择，decoder 按显式 indices 解码：

```cpp
VideoInfo inspect_video(std::span<const std::uint8_t> bytes,
                        const Policy& policy);

Video decode_video(std::span<const std::uint8_t> bytes,
                   const Policy& policy,
                   std::span<const int> frame_indices);
```

VideoInfo 提供真实 source 几何、frame count、fps/duration；不含某个模型的
sampled_frames 或预先选择的 indices。传入 indices 非递减、在 source 范围内，
允许末帧重复；decoder 可以一次解码相同源帧，再按计划交付，不自行换采样方案。
Decoder 拥有 crop/orientation、解码失败、物理内容有效性与媒体资源限制。

把现有 Qwen sampler 移到 Qwen processor，保持其选择结果；GLM 实现自己的 sampler。
移除被取代的采样式 decoder 接口，不留一套只转发的新旧并存路径。
inspect 与 decode 使用一致的帧编号和显示几何，不能仅依赖可能不准确的容器帧数
生成实际无法解码的选择。

CPU 解码/resize/patchify 复用现有 CPU 执行基础及媒体任务管理，不建立新线程池。
这是媒体准备；Vision 神经网络计算在 GPU 完成。

### 7.3 Token 与 Main 输入

图像按模型 image-start/end 和 image placeholder 展开；视频按 video-start/end，
内部每个 temporal grid 的 image block 与 timestamp 文本展开。Token IDs 从
artifact/config 读取。GLM 参考时间文本为一位小数的 `seconds` 表述，字符串构造
属于 Frontend，不能由 GPU Vision 代码补写。

一个 grid 的 feature 数为 Hg*Wg/4，必须与对应 placeholder span 一一对应。
Main embedding lookup 后，在进入 mHC expand 前 scatter 替换这些位置。
请求 chunk 只消费其 token 窗口相交的 feature spans。

MTP 的 e 采用词表 embedding，见[MTP 输入合同](mtp.md#21-main-和-mtp-的位置)；
不将 Qwen 的媒体 embedding 对齐逻辑直接复用到 GLM MTP。

## 8. 双机任务划分

### 8.1 权重与工作单位

两 rank 各加载完整 BF16 Vision tower；精确 payload 约 1.049837 GiB/rank，
见数值参考。第一版不在 tower 内做 TP。

分配单位是完整 attention segment，即一个 temporal grid。单张静态图像只有一个
segment；视频的多个 temporal grids 可以分到不同 ranks，因为它们在 attention、
downsample 和 merger 中都没有跨段依赖。单个 segment 不沿 N 切开后独立做 attention。

每个 rank 对自己的 segments 执行完整 tower；内部可以 pack 多个段，边界使用
cu_seqlens。中央模型计划冻结：

```text
(item, temporal_grid, patch_range, feature_range, token_span, destination_rank)
```

该映射派生自准备好的媒体数据，rank 不重新选择媒体身份或分配。初始分配使用
由实际几何推导的投影工作量与段内 attention 工作量，做确定性的负载分配；
不要求请求的 MLA owner 同时执行其 Vision。

### 8.2 输出交接

两侧完成本地 features 后，按已冻结的原始 feature/token 顺序交换必要结果，
使后续复制的 Main residual 处理相同输入。交换只位于 Vision→Main 的边界，
不在 24 层内部增加 collectives。

单图只在一个 rank 执行，仍需将最终 features 交给另一 rank；多图/视频可以并行。
执行顺序和网络到达顺序不能改变媒体 token 的顺序。重复图像是否复用 prepared
payload 由已有媒体缓存决定，不等于重复投放同一份 GPU 计算结果。

传输优先发送最终 [4096,Nmerged]，不交换 encoder 中间激活。
复制部署的容量与通信预算由 [parallel.md](parallel.md#61-当前量化方案的权重预算)
统一维护。Op 接口不增加 communicator 或 request owner 参数。

## 9. 资源生命周期、缓存与 Graph

| 资源 | 所有者与释放点 |
|---|---|
| Encoded media／CPU preparation | Frontend/media task；准备完成后按已有缓存政策释放或保留 |
| Prepared BF16 patches | 不可变 payload；最后上传/消费者完成后释放引用 |
| Vision 权重 | ModelShard；加载期一次物化 |
| Device positions、cu_seqlens、patch staging | RankProgram；一个有界 segment batch 内复用 |
| Encoder、FFN、merger activations | RankProgram frame；按层和阶段复用 |
| 最终 features | 保留到对应 Main prefill spans 的最后消费者完成 |
| Communication buffers | 分布式 program；依赖覆盖发送和接收完成 |

Workspace 按执行中的 segment batch 分配，不把整个最大视频的中间激活一次性常驻。
单 segment 的 attention 必须看完整段，分块计算由 attention Op 内部保持完整数学，
模型不能用截短序列来满足较小 workspace。

生成期只需要已建立的 Main context，不要求保存原 Vision 激活。Prefix 命中后的
媒体计算需求由恢复点与剩余 Main 输入 spans 决定；已覆盖的媒体无需重复消费
features，未覆盖部分按既有媒体计划准备。跨 rank 的上下文恢复仍由 Session 统一处理。

本阶段不新增长期 GPU feature-cache 策略。现有 CPU prepared-media cache、
singleflight、取消检查和统计沿用共同机制，cache key 中保留 GLM processor 的真实身份。
活动媒体 feature 生命周期不应错误地挂在整个生成请求上。

Vision 在 prefill 前或期间执行，不改变 decode 的完整 Graph 组织。固定形状的数学
Ops 均满足捕获要求，但无需为本轮实现单独的 Vision 模型 Graph 或 prefill Graph。

## 10. 文件组织与实施步骤

```text
src/media/decode/                    通用 probe 与按 indices 解码
src/models/common/frontend/         共享 owning media 数据、缓存机制
src/models/glm5_next/
    frontend/                       GLM processor、媒体展开、token/position 语义
    vision/                         参数视图、encoder/merger 组合、执行资源
    parallel/                       segment placement 与输出交接
include/ninfer/ops/
    linear_bias.h
    linear_split.h                  bias 形式
    linear_swiglu.h                  bias clamped 形式
    rmsnorm_rope.h                   2D norm+RoPE
    rope.h / softmax_attention.h     D64 支持
src/ops/
    linear/bias/
    linear/split/
    linear/swiglu/
    rope/
    softmax_attention/
```

简单的接口扩展留在原 family；复杂路线按既有 device/format/shape 组织。
Converter 的 patch/downsample 重排和 QKV/GU bias 合并放在 GLM recipe，
不在运行时临时整理权重。

1. **输入与布局**：完成共享 media 数据边界、显式 frame 解码、GLM processor、
   token spans，以及 converter 的保值布局。用小图和短视频独立验证。
2. **计算支持**：在纯 Linear 基础上完成 bias/split/clamped 形式、共享 residual/norm、
   D64 norm+RoPE 与 dense/packed attention。各项可以在接口确定后并行实施。
3. **Vision 组合**：先接通完整 encoder/downsample/merger 与 Main embedding 替换，
   随后接入已定 segment placement 和结果交换，复用同一执行组合。
4. **模型与服务收尾**：覆盖图像、视频、混合文本、prefix restore、取消及 MTP 共用；
   集中检查真实媒体的功能与性能。

## 11. 验证与调优范围

正确性使用独立数学和小型可读样例：

- 预处理检查对齐 canvas、缩放内容、归一化 padding、frame 选择/重复、
  timestamps、patch feature 和列次序；预算估计与实际物化一致。
- Patch projection 与原 Conv3d 数学对应，downsample view＋重排矩阵与原 Conv2d
  对应；两者均检查 bias。使用可区分 channel/tap/位置的输入避免错误排列偶然相消。
- 检查 bias 在 clamp 前、gate 单侧 clamp、direct gamma、Exact GELU 和 BF16
  residual 边界；不把内部更高精度作为测试要求。
- D64 norm/RoPE 分别验证坐标轴、split-half 方向与频率；packed attention 使用
  多段不同长度，确认没有跨段读入和尾 tile 丢失。
- Features 与展开 token spans、双机乱序完成后的重组、图像/视频混合输入及缓存恢复
  使用同一 PreparedPrompt 检查。

调优重点按数据规模而不是文本 token 长度选择。图像 8000 个 merged tokens 对应
最多约 32000 raw patches；纯 Linear 的 2048/4096 结果不代表完整 Vision 域已覆盖。
至少包含小图、典型分辨率、高分辨率以及多 temporal grids 的 packed 场景，
资格测试用代表边界即可，不建立庞大笛卡尔积。

分别记录媒体准备、encoder、downsample/merger、结果交换和 Main prefill 时间。
对于长度 N_s 的独立 segments，attention 主矩阵计算量按
`4*heads*head_dim*sum_s(N_s²)`／层计，不用整个 packed N 的平方。
Projection 按实际矩阵尺寸和 raw/merged 列数计 FLOPs；带宽口径包含实际权重、
activation 和 scratch 流量，并展示 roofline。只有完整模型测量才说明媒体 TTFT 收益。

## 12. 参考入口

- [模型数学](../../maintainer/glm5next-model.md#62-vision)、
  [数值政策](../../maintainer/glm5next-quantization.md)、
  [权重布局](parallel.md#21-vision-权重的保值布局)、[实施路线](roadmap.md)。
- 本地 vLLM：`vllm/models/glm5next/common/multimodal.py` 的实际 tower、
  `vllm/transformers_utils/processors/glm5next.py` 的 resize/pad/patchify/frame sampler，
  以及 `vllm/model_executor/models/glm4_1v.py` 中 GLM5Next 复用的媒体展开。
- Checkpoint 的 config/processor 配置优先于参考代码中的默认参数或旧注释；
  这里不沿用默认构造器中不适用于当前实例的 norm epsilon。
