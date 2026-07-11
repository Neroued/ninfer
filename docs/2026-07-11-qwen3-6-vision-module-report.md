# Qwen3.6-27B Vision 模块调研报告

> 日期：2026-07-11  
> 范围：本地 `Qwen3.6-27B/base-hf-bf16` checkpoint、Hugging Face `Qwen3_5` 实现、
> Qwen3-VL processor，以及本项目当前 q5090/L0/L1/L2/serve 支持情况。  
> 本文只描述模型语义、输入协议、计算图、算子需求和工程缺口，不是正式实现计划。

## 1. 核心结论

当前 checkpoint 的 Vision 路径可以概括为：

```text
图片/视频
  ↓ 解码、缩放、归一化、patchify
视觉 patch [P,1536]
  ↓ 27 层 Vision Transformer
视觉 hidden [P,1152]
  ↓ 2×2 Patch Merger
视觉 embedding [P/4,5120]
  ↓ 覆盖 <|image_pad|>/<|video_pad|> 的文本 embedding
Qwen3.6 语言模型 prefill
  ↓
正常文本生成
```

关键结论如下：

1. 本地所谓的 `Qwen3.6-27B`，架构标识仍然是 `qwen3_5`，由
   `Qwen3_5ForConditionalGeneration` 加载。官方 Transformers 文档也明确说明 Qwen3.6 dense
   checkpoint 和 Qwen3.5 共用模型类型和实现。
2. Vision 不是 cross-attention。视觉塔先生成 5120 维 embedding，直接替换文本序列里视觉
   placeholder 的词向量，随后所有视觉和文本 token 一起经过现有 64 层语言模型。
3. 该 checkpoint 原生内容模态只有文本、图片和视频。没有音频 encoder。虽然词表中存在
   audio/TTS 特殊 token，通用 processor 接口也保留了 `audio` 参数，但这个 checkpoint 的模型
   执行路径只有 `image` 和 `video`。
4. 图片和视频共享同一个视觉塔。视频额外涉及抽帧、每两个相邻帧组成一个 temporal patch、
   文本时间戳，以及语言模型侧的 MRoPE。
5. 当前工程已经完整保存和加载 Vision 权重，但还没有实际计算路径。现状是“格式就绪、运行时
   未接入”。

官方依据：

- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/README.md)
- [Transformers Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

本地 checkpoint 依据：

- `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/config.json`
- `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/preprocessor_config.json`
- `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/video_preprocessor_config.json`
- `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/chat_template.jinja`

## 2. 模型支持的内容

### 2.1 内容模态

| 类型 | 是否支持 | 说明 |
|---|---:|---|
| 普通文本 | 是 | 和当前 text-only 路径一致 |
| 单张图片 | 是 | 动态分辨率 |
| 多张图片 | 是 | 可与文本任意交错 |
| 视频 | 是 | 抽帧后处理，附带文本时间戳 |
| 多段视频 | 是 | 每段视频有独立 grid 和时间信息 |
| 图片与视频混合 | 是 | 按消息中的出现顺序处理 |
| PDF | 非原生 | 需要先把页面渲染成图片 |
| GIF/动画 | 非原生 | 可由前端按图片或视频解码 |
| 音频 | 否 | 没有 audio encoder |
| 输出图片/视频 | 否 | 模型只生成文本 token |

图片可以表示照片、截图、扫描件、表格、图表、UI、数学题和文档页面。OCR、grounding、框坐标
和 GUI 操作属于文本输出能力，不是额外输入模态。官方评测覆盖 VQA、OCR、文档理解、空间定位、
视频理解和视觉 Agent 等任务，见
[Qwen3.5-27B Vision Language 评测](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/README.md#vision-language)。

需要区分媒体传输格式与模型模态：

- 图片 URL、本地路径、base64、PIL、NumPy 和 Tensor 是前端传输或解码形式；
- 对模型而言，它们最终都是 RGB 像素；
- 官方 `qwen-vl-utils` 支持本地路径、HTTP(S)、`file://`、base64 data URI 和 PIL Image，见
  [官方视觉输入处理实现](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)。

### 2.2 Chat 内容结构

本地 chat template 接受交错内容：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "比较这两张图片："},
    {"type": "image", "image": "..."},
    {"type": "image_url", "image_url": "..."},
    {"type": "text", "text": "有什么不同？"}
  ]
}
```

约束：

- system/developer 消息不能包含图片或视频；
- user 消息可以任意交错 text/image/video；
- 可以启用 `add_vision_id`，在视觉内容前生成 `Picture 1:`、`Video 1:`；
- 多模态内容可以与 thinking、tool calling 共存。

图片模板最初生成：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

processor 根据实际分辨率，把单个 `<|image_pad|>` 扩展为正确数量：

```text
<|vision_start|>
<|image_pad|> × V
<|vision_end|>
```

相关 token ID：

| token | ID |
|---|---:|
| `<|vision_start|>` | 248053 |
| `<|vision_end|>` | 248054 |
| `<|vision_pad|>` | 248055 |
| `<|image_pad|>` | 248056 |
| `<|video_pad|>` | 248057 |

processor 还生成 `mm_token_type_ids`：

- `0`：文本、时间戳、vision start/end 等普通 token；
- `1`：`image_pad`；
- `2`：`video_pad`。

只有 pad token 对应的文本 embedding 会被视觉 embedding 替换。

## 3. 图片预处理协议

### 3.1 解码与归一化

输入首先：

1. 转为 RGB；
2. bicubic resize；
3. 除以 255；
4. 使用 `mean=[0.5,0.5,0.5]`、`std=[0.5,0.5,0.5]` 归一化。

公式：

```text
x = (pixel / 255 - 0.5) / 0.5
```

通常落在 `[-1,1]`。

### 3.2 动态分辨率

关键参数：

```text
patch_size          = 16
spatial_merge_size  = 2
resize factor       = 16 × 2 = 32
```

resize 后的高、宽都必须是 32 的倍数，同时尽量保持原始宽高比。本地配置的像素面积范围：

```text
min_pixels = 65,536
max_pixels = 16,777,216
```

宽高比不能超过 200。`shortest_edge/longest_edge` 的名字容易误导：在 Qwen processor 中它们
实际参与的是总像素面积预算，不是单条边长度。

### 3.3 Temporal patch

视觉 patch embedding 是：

```text
Conv3d:
  kernel = [2,16,16]
  stride = [2,16,16]
  input channels = 3
  output channels = 1152
```

单张图片只有一帧，processor 会把它复制成两帧，构成一个 temporal patch：

```text
3 channels × 2 frames × 16 × 16 = 1536
```

最终每个 patch 是一个 1536 维向量。q5090 已经把 Conv3d 权重展平成 `[1152,1536]`，运行时
不需要实现通用 Conv3d，可以直接调用 `linear()`。权重展平顺序固定为 `C,T,H,W`，W 变化最快，
见 [q5090 格式规范](q5090_packed_file_format_v4.md#14-layout-assignment-policy)。

### 3.4 Patch 排列

processor 不是简单的 row-major patch 排列，而是先按 2×2 spatial merge block 排列：

```text
[T, H/2, W/2, merge_h=2, merge_w=2, C, temporal=2, patch_h=16, patch_w=16]
```

这样每连续四个视觉 patch 正好属于同一个 2×2 区域，后面的 merger 可以直接：

```text
[P,1152] → [P/4,4608]
```

而不需要额外 gather。

### 3.5 Token 数公式

设 resize 后尺寸为 `Hr × Wr`：

```text
H = Hr / 16
W = Wr / 16

ViT patch 数 P = H × W
LLM 视觉 token 数 V = P / 4
```

本地真实 processor 结果：

| 原始图片 | resize 后 grid | ViT patch P | LLM token V |
|---|---:|---:|---:|
| 100×100 | 16×16 | 256 | 64 |
| 224×224 | 16×16 | 256 | 64 |
| 1024×1024 | 64×64 | 4096 | 1024 |
| 1920×1080 | 120×68 | 8160 | 2040 |
| 4000×4000 | 250×250 | 62500 | 15625 |

配置上单图最多约：

```text
P <= 65,536
V <= 16,384
```

## 4. 视频预处理协议

### 4.1 抽帧

默认参数：

```text
target fps = 2
min_frames = 4
max_frames = 768
temporal_patch_size = 2
```

processor 根据视频原始 FPS 和时长均匀采样。若帧数为奇数，复制最后一帧补成偶数。

如果缺少视频 metadata，processor 会退回假定原视频是 24 FPS，但这会导致时间戳不准确。因此
视频输入需要携带：

- 原始 FPS；
- 原始总帧数；
- 实际采样 frame indices。

### 4.2 视频 grid

对抽样后的 `F` 帧：

```text
T = ceil(F / 2)
H = resized_height / 16
W = resized_width / 16

ViT patch P = T × H × W
LLM video token V = P / 4
```

每两个相邻采样帧被同一个 3D patch embedding 联合处理。

### 4.3 文本时间戳

Qwen3.6 不依靠 ViT 内部的 temporal RoPE 建立完整时间轴，而是把每个 temporal patch 对应的
时间写成文本：

```text
<0.5 seconds>
<|vision_start|>
<|video_pad|> × frame_seqlen
<|vision_end|>
```

时间戳取两帧实际时间的平均值。整体类似：

```text
<|vision_start|>
  <0.5 seconds>
  <|vision_start|> video tokens for frames 0/1 <|vision_end|>
  <1.5 seconds>
  <|vision_start|> video tokens for frames 2/3 <|vision_end|>
  ...
<|vision_end|>
```

这是 Qwen3-VL/Qwen3.5 相比旧式 T-RoPE 的重要变化：视频事件时间通过正常文本 token 显式
进入语言模型。官方架构将其称为 Text–Timestamp Alignment，见
[Qwen3-VL 架构说明](https://github.com/QwenLM/Qwen3-VL#model-architecture-updates)。

## 5. Vision Transformer 的精确结构

本地视觉塔共有：

```text
333 tensors
460,730,096 parameters
BF16 原始大小约 878.77 MiB
q5090 v4.2 大小约 282.02 MiB
```

配置：

```text
depth                    = 27
hidden_size              = 1152
intermediate_size        = 4304
num_heads                = 16
head_dim                 = 72
patch_size               = 16
temporal_patch_size      = 2
spatial_merge_size       = 2
out_hidden_size          = 5120
num_position_embeddings  = 2304 = 48 × 48
```

具体权重和量化格式固定在
[`build_vision_specs()`](../tools/q5090_convert/tensor_plan.py#L1139)。

### 5.1 Patch embedding

```text
input:  [P,1536]
weight: [1152,1536] Q6
bias:   [1152] BF16
output: [P,1152]
```

计算：

```text
x = linear(patches, patch_weight)
x += patch_bias
```

### 5.2 可学习绝对位置编码

位置表：

```text
pos_embed.weight = [2304,1152] = [48×48,1152]
```

输入分辨率动态变化，因此不是直接索引，而是把目标 `H×W` grid 映射到 48×48 表上，做双线性
插值：

```text
position = Σ four_corner_weight × pos_embed[four_corner_index]
x += position
```

插值结果按 2×2 merge-friendly 顺序重排，并沿 T 维重复。

### 5.3 Vision RoPE

Vision attention 使用二维 RoPE：

- head dimension 为 72；
- 所有 72 维都参与旋转；
- position 只有 `(row,column)`；
- 不包含 temporal position；
- theta 为 10000。

每个 patch 的二维 position 按 merge-friendly 顺序生成。

### 5.4 27 个 Transformer block

每层：

```text
h = LayerNorm(x)
qkv = linear(h, qkv_weight) + qkv_bias
q, k, v = split(qkv)
q, k = vision_rope(q, k)
a = noncausal_attention(q, k, v)
a = linear(a, proj_weight) + proj_bias
x = x + a

h = LayerNorm(x)
h = linear(h, fc1_weight) + fc1_bias
h = GELU_tanh(h)
h = linear(h, fc2_weight) + fc2_bias
x = x + h
```

形状：

| 计算 | 权重 | 格式 |
|---|---:|---|
| QKV | `[3456,1152]` | Q4 |
| attention proj | `[1152,1152]` | Q5 |
| MLP fc1 | `[4304,1152]` | Q4 |
| MLP fc2 | `[1152,4304]` | Q5 |
| norm weights/biases | `[1152]` | BF16 |

这里是标准 affine LayerNorm，不是当前文本模型的 RMSNorm：

```text
y = (x - mean(x)) / sqrt(var(x) + 1e-6) × weight + bias
```

Vision block 的激活由 `gelu_pytorch_tanh` 指定，是 tanh 近似 GELU。

### 5.5 Vision attention 的边界

Vision attention 是：

- 16 个 Q/K/V heads；
- head dimension 72；
- 非因果；
- 无 KV cache；
- 每个序列段独立 full self-attention。

对于 grid `[T,H,W]`，`cu_seqlens` 把它拆成 T 个长度为 `H×W` 的独立 attention 段：

```text
segment 0: temporal patch 0 的 H×W patches
segment 1: temporal patch 1 的 H×W patches
...
```

因此：

- 不同图片之间不互相 attention；
- 不同视频 temporal patch 之间不互相 attention；
- ViT 内的时间混合仅来自最开始的两帧 Conv3d patch；
- 更长时间范围的关系由时间戳文本和语言模型处理。

视频 Vision attention 复杂度是：

```text
T × (H×W)²
```

而不是：

```text
(T×H×W)²
```

### 5.6 Patch merger

27 层之后：

```text
x = LayerNorm(x)                   # 仍在 [P,1152] 上
x = view(x, [P/4,4608])            # 四个相邻 patch 拼接
x = linear(x, fc1) + bias          # [4608,4608], W8G32
x = GELU_exact(x)
x = linear(x, fc2) + bias          # [5120,4608], W8G32
```

输出：

```text
[V=P/4,5120]
```

merger 使用 `nn.GELU()` 默认精确版本，不是 block 中的 tanh 近似版本。实现时必须保留两种
GELU 路径。

本地 checkpoint 的 `deepstack_visual_indexes=[]`，没有 Qwen3-VL 某些变体中的多层 DeepStack
注入。这里只使用最终 merger 输出。

## 6. 视觉 embedding 如何进入语言模型

先正常执行文本 embedding：

```text
inputs_embeds = embed_tokens[input_ids]
```

然后找到：

```text
input_ids == image_token_id
input_ids == video_token_id
```

并检查 placeholder 数和视觉特征数严格相等，随后：

```text
inputs_embeds[image_mask] = image_embeds
inputs_embeds[video_mask] = video_embeds
```

`vision_start`、`vision_end`、时间戳和普通文本仍使用正常词向量。

完成替换后，不再有独立的 Vision/Language 交互模块。整个 mixed sequence 直接进入现有语言
模型：

```text
64 × (
  3 个 GDN layer
  1 个 full-attention layer
)
```

Vision 只参与 prefill：

- decode 阶段不会重新运行视觉塔；
- 视觉 embedding 已经被折叠进 GDN state、KV cache 和上下文 hidden；
- Vision 临时 activation 可以在 prefill 后释放；
- 如果请求稀疏，可以考虑在视觉 prefill 后释放或换出约 282 MiB 的 Vision 权重。

## 7. 语言模型的 MRoPE

这是接入 Vision 时最容易漏掉、同时会造成明显错误的部分。

当前 text-only RoPE 使用单一 position。多模态时 position 变成：

```text
position_ids: [3,T]
axes = temporal, height, width
```

### 7.1 文本 token

普通文本的三个轴相同：

```text
t = h = w = current_position
```

因此退化为现有 1D RoPE。

### 7.2 图片 token

对 merger 后的视觉 grid `[1,H/2,W/2]`，position 为：

```text
temporal = image_start_position
height   = image_start_position + row
width    = image_start_position + column
```

视觉区域结束后，逻辑 position 只前进：

```text
max(H/2,W/2)
```

而不是前进 `H×W/4`。所以长视觉 token 区域在 MRoPE 空间中被压缩。

### 7.3 视频 token

视频的每个 temporal patch 被时间戳和 vision wrapper 分隔，然后按一个独立的 `[1,H,W]` grid
处理。时间关系主要通过：

- `<x.x seconds>` 文本；
- 不同视觉段之前累积的文本 position；
- 后续语言模型 full attention/GDN state；

进行编码。

### 7.4 Interleaved 频率分配

文本 full-attention 的 rotary dimension 是 64，即 32 个频率 pair。

```text
mrope_section = [11,11,10]
```

实际不是连续的 `TTT...HHH...WWW...`，而是交错分配：

```text
pair index j:
  j % 3 == 0 → temporal，11 个
  j % 3 == 1 → height，11 个
  j % 3 == 2 → width，10 个
```

当前 [`rope()`](../include/qus/kernels/rope.h) 和实现固定为：

- 1D positions；
- head dimension 256；
- Q heads 24；
- K heads 4。

因此不能直接支持 Vision RoPE 或文本 MRoPE，需要扩展。

prefill 结束还需要保存：

```text
rope_delta = max(multimodal_position) + 1 - prompt_token_count
```

后续 decode token 的三个 position 都是：

```text
ordinary_decode_position + rope_delta
```

如果不保存这个 delta，首轮视觉回答可能看似可用，但长 decode 和多轮追加会逐步错位。

## 8. 所需算子

### 8.1 可直接复用

| 功能 | 当前能力 | 结论 |
|---|---|---|
| 所有视觉线性层 | `linear()` | 可复用 |
| Q4/Q5/Q6 | 已支持 generic shape | 可复用 |
| merger W8G32 | 已支持 large-T MMA | 可复用 |
| residual add | 已有 | 可复用 |
| 文本 embedding gather | 已有 | 可复用 |
| 语言模型 GDN/full attention | 已有 | 主体可复用 |

现有 `linear()` 的输入 `[K,T]` 内存布局对应 processor 的 `[T,K]` row-major buffer，因此只需
创建 Tensor view，不需要转置数据。

### 8.2 必须新增或扩展

| 建议 API | 功能 | 关键要求 |
|---|---|---|
| `layer_norm()` | Vision affine LayerNorm | FP32 mean/variance，BF16 输出，weight+bias |
| `add_bias()` | 按 channel 广播 bias | 支持 `[N,T] + [N]` |
| `gelu()` | 两种 GELU | tanh approximate / exact |
| `vision_pos_embed()` | 四点双线性 embedding gather+add | 动态 H/W，merge-friendly 排列 |
| `rope()` 扩展 | 二维 Vision RoPE 与三维 text MRoPE | 内部按 position 维度和固定模型形状分派 |
| `vision_attention()` | packed 非因果 MHA | 16 heads、D=72、`cu_seqlens`、无 cache |
| `scatter()` | 把视觉 embedding 写入 placeholder | 严格检查 token/feature 数一致 |
| position builder | 生成 token types、三维 positions、rope delta | 可放 CPU，成本很小 |

### 8.3 不需要实现

- 通用 Conv3d：patch 已经展开为线性层；
- cross-attention；
- Vision KV cache；
- causal Vision attention；
- window attention；
- DeepStack；
- 通用动态 ViT 框架；
- 音频算子。

### 8.4 Fused op

为了保持顶层 API 语义清晰，不应让 `linear()` 隐式处理 bias 或 activation。

正确性阶段：

```text
linear()
add_bias()
gelu()
```

性能确实需要时，再明确引入改变语义的 fused API，例如：

```text
linear_bias()
linear_bias_gelu()
linear_bias_add()
```

这样 `linear()` 始终只表示矩阵乘法，而 fused op 名字清楚表达额外含义。

`vision_attention()` 也不应塞进现有 `gqa_attention()`：

- `gqa_attention()` 是 causal GQA，24Q/4KV、D=256、带 KV cache；
- Vision 是 noncausal MHA，16Q/16K/16V、D=72、packed variable-length；
- 二者数学语义不同，不应仅靠 shape 暗中分派。

## 9. 当前工程状态

### 9.1 已完成

1. q5090 v4.2 已包含完整 `VISION_ENCODER`：
   - 333 tensors；
   - 54 个 Q4 block；
   - 54 个 Q5 block；
   - 1 个 Q6 patch weight；
   - 2 个 W8G32 merger weight；
   - 222 个 BF16 tensor。
2. Vision 量化格式和形状锁已经完成，见
   [q5090 格式策略](q5090_packed_file_format_v4.md#14-layout-assignment-policy)。
3. `WeightStore` 已支持选择性加载 Vision，见
   [`LoadOptions`](../include/qus/core/weight_store.h#L19)。
4. 当前真实 artifact 中 Vision payload 为：

```text
295,719,424 bytes ≈ 282.02 MiB
```

### 9.2 尚未完成

1. `Qwen3_6_27B` 没有 Vision 权重绑定和视觉 forward。
2. 模型 prefill 只接受 token IDs：

```cpp
prefill(std::span<const int> ids)
```

无法接收：

- 已替换的 input embeddings；
- 三轴 position IDs；
- rope delta；
- 图片/视频 grid metadata。

见 [`Qwen3_6_27B`](../include/qus/model/model.h#L135)。

3. 当前 RoPE 只支持 text-only 1D 路径。
4. OpenAI schema 识别 `image_url`，但没有保存 URL/payload 内容，并在 translate 阶段拒绝，见
   [`to_chat_messages()`](../src/serve/translate.cpp#L63)。
5. Anthropic image block 直接返回 `modality_not_supported`。
6. 当前 request DTO 没有 image/video source、MIME type、decoded media ownership 或 per-media
   preprocessing options。
7. 当前 server context 上限仍是 8192，而一张 1080p 图片已经消耗约 2040 个 LLM token。

## 10. 性能与预算

### 10.1 不能只有 LLM token 预算

Vision merger 后的 token 数为：

```text
V = T×H×W/4
```

但 ViT attention 在 merger 前执行，计算量取决于：

```text
A = T×(H×W)²
```

admission control 至少需要三个预算：

```text
raw_patch_budget         = Σ T×H×W
llm_vision_token_budget  = Σ T×H×W/4
vision_attention_budget  = Σ T×(H×W)²
```

只检查最终 context token 数会允许一些“LLM context 合法、ViT 计算极其昂贵”的图片。

例如 4000×4000 图片：

```text
ViT patches = 62,500
LLM tokens = 15,625
attention pairs per layer = 3.906 billion
```

即使 FlashAttention 不存完整 attention matrix，计算量仍然存在。

### 10.2 必须使用 FlashAttention 风格实现

不能物化 `[16,P,P]` attention score。以 1080p 为例：

```text
P = 8160
16 × P² × 2 bytes ≈ 1.98 GiB
```

这还只是单层 BF16 score。因此 `vision_attention()` 必须：

- 使用 online softmax；
- 使用 FP32 max/sum accumulation；
- 不写出完整 score matrix；
- 支持 packed `cu_seqlens`；
- 分段禁止跨图或跨 temporal patch attention。

### 10.3 推荐生命周期

```text
加载 Vision 权重
→ 处理全部视觉资产
→ 得到 [V,5120] embedding
→ 释放 Vision activation
→ scatter 到 text embedding
→ text prefill
→ 可选卸载 Vision 权重
→ decode
```

高频多模态服务可以常驻约 282 MiB Vision 权重；低频多模态请求可以延迟加载，但需要权衡约
282 MiB H2D 上传时间。

多轮对话中，如果历史图片没有变化，最好缓存最终 5120 维视觉 embedding 或完整 prefix state，
而不是重新运行视觉塔。

## 11. 推荐实现顺序

### 阶段 1：复刻 processor

先在 CPU 侧精确实现：

- 图片/视频内容解析；
- RGB decode；
- bicubic resize；
- normalize；
- frame sampling 和 metadata；
- patchify/reorder；
- placeholder expansion；
- `mm_token_type_ids`；
- Vision/Text position IDs。

用少量 canonical fixture 与 Hugging Face processor 比较：

- resize 后尺寸；
- `grid_thw`；
- patch buffer；
- placeholder token 数；
- token type runs；
- 三轴 position IDs；
- video timestamp。

### 阶段 2：基础 Vision 算子

实现：

- `layer_norm()`；
- `add_bias()`；
- `gelu()`；
- `vision_pos_embed()`；
- Vision 二维 `rope()`；
- `vision_attention()`。

先用 BF16 或解量化 oracle 做单层数值验证。

### 阶段 3：完整 Vision tower

建立静态 `Qwen3_6_Vision` 模型卡：

- 绑定 333 个权重；
- 复用 `linear()`；
- 运行 27 层 block；
- 运行 merger；
- 输出 `[V,5120]`。

至少验证 patch embedding、第 1/中间/第 27 层 hidden 和 merger 输出。

### 阶段 4：语言模型接入

改造 prefill，使其支持：

- token IDs；
- optional visual embeddings；
- placeholder scatter；
- `[3,T]` MRoPE position；
- rope delta 保存；
- decode 延续。

GDN 层无需视觉专用修改；只需 full-attention RoPE 使用正确的 MRoPE。

### 阶段 5：图片 E2E

先支持：

- 单图；
- 多图；
- 图文交错；
- thinking on/off；
- tool calling + image。

完成后再接视频，避免同时调试抽帧、时间戳和模型计算。

### 阶段 6：视频和服务协议

最后加入：

- 视频 decoder；
- metadata；
- 2 FPS/default frame sampling；
- timestamp prompt；
- per-request vision budget；
- OpenAI/Anthropic multimodal schema；
- URL/base64 安全限制。

## 12. 最终判断

Vision 的权重格式已经准备充分，下一阶段不需要再改 q5090。真正的工作量集中在：

1. processor 和输入协议；
2. noncausal packed Vision attention；
3. LayerNorm、GELU、位置插值等视觉辅助算子；
4. 文本三维 interleaved MRoPE；
5. input embedding scatter 和模型接口；
6. 服务侧媒体输入与双重预算。

其中风险最高的部分是：

- patch 排列必须与 merger 完全一致；
- Vision attention 必须按 temporal patch 分段；
- block GELU 和 merger GELU 不能混淆；
- placeholder 数必须严格匹配；
- text MRoPE 和 decode rope delta 必须正确；
- 预算必须基于 merger 前的二次 attention 复杂度。

这套结构适合当前工程：绝大多数参数计算都可以复用统一的 `linear()`，新增算子数量有限，也不
需要引入通用多模态框架。

