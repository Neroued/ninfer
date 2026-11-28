# Inkling2 架构

Inkling-2-Mini 使用 `Inkling2ForCausalLM`（`inkling2_text`）：Looped Block-AttnRes 主干、短卷积相对位置混合注意力、软截断 NVFP4_MUL/MXFP8 MoE、PE-Lang Vision 与 DSpark 投机解码的组合。非循环（`Inkling2For*`）与循环（`Inkling2LoopFor*`）checkpoint 统一映射到同一模型入口，维度与循环拓扑全部由 artifact config 驱动。主要入口为 [config](../../src/models/inkling2/config.cpp)、[load](../../src/models/inkling2/load/)、[execution](../../src/models/inkling2/execution/)、[program](../../src/models/inkling2/program/)、[state](../../src/models/inkling2/state/context.cpp) 与 [frontend](../../src/models/inkling2/frontend/)。

## 主干、循环核心与 Block-AttnRes

```text
embedding → embed_norm (+ Vision features) → Block-AttnRes stack[0]
  → Prelude (layers 0..5, 1 pass)
  → Looped Core (layers 6..17 × 3 passes, rank-64 message gates)
  → Coda (layers 18..23, 1 pass)
  → final_norm → output_head → logit_softcap (30.0)
```

Inkling-2-Mini 由 24 个物理层（`hidden=2304`）展开为 48 次逻辑层访问。每 6 层构成一个 Block-AttnRes 块（`n_layers_per_attn_res_block=6`，`attn_res_normalize_values=true`）。每层注意力与 MLP 输出均先经 4-tap 因果短卷积（`sconv_kernel_size=4`）再汇入块内局部残差和并参与下一次 Block-AttnRes 聚合；正常路径通过 `short_conv_block_attn_res` 融合执行，投机 verify 路径使用 `short_conv_block_attn_res_snapshot` 写出紧凑卷积快照供提交阶段按真实接受前缀折叠。

循环核心在 `pass_index == 0` 入口保存 `prelude_partial`；进入 `pass_index > 0` 时，将上一轮核心生成的块栈（`bank_blocks=2`）复制到 `roots_.bank`，恢复 `prelude_partial`，再用各层 `message_queries` 经 `block_attn_res` 提取跨轮消息，并通过 `loop_message_blend_rmsnorm` 融合到注意力输入与 MLP 输入中。实现见 [loop_residual](../../src/models/inkling2/execution/loop_residual.cpp) 与 [text](../../src/models/inkling2/execution/text.cpp)。

## 短卷积相对注意力、KV 与 State

物理层包含 20 层局部 SWA（`sliding_window_size=1024`，展开为 40 次逻辑访问）与 4 层全局注意力（物理层 `[5, 11, 17, 23]`，展开为 8 次逻辑访问），均为 24 Q / 8 KV heads（`head_dim=128`）。注意力具备：

- K/V 因果短卷积（`k=4`，全局层额外带 `k_conv_proj` / `v_conv_proj`）；
- 学习型相对位置偏置（`d_rel=16`，`rel_extent=1024`）及 Q/K/Rel L2 裁剪；
- 无 gamma 的 Value RMSNorm、输出门控（output gate）、attention sink 与长上下文 log scaling（`n_floor=8192`，`alpha=0.1`）。

状态与缓存布局由 [state/context](../../src/models/inkling2/state/context.cpp) 和 [program/planning](../../src/models/inkling2/program/planning.cpp) 统一规划：

| 存储类别 | 位置 | 表示与说明 |
|---|---|---|
| Main 全局注意力 KV | Paged KV Pool 0 | `kv_dtype`：`bf16` 或 `fp8`（E4M3FN group-128 UE8M0） |
| Main 局部 SWA KV（40 逻辑层，`W=1024`） | `StateImage` cyclic KV | `swa_kv_dtype`：`bf16`（默认）或 `fp8`（高并发省显存） |
| K/V 与 Attn/MLP 短卷积历史（3 taps） | `StateImage` | BF16，按逻辑层保存 |
| `final_hidden`（及可选 `dspark_f_rnn`） | `StateImage` | BF16，用于续算与 DSpark 状态延续 |
| DSpark 全局层（layer 1）KV | Paged KV Pool 1 | BF16 paged KV |
| DSpark 局部层（layers 0, 2）KV | `StateImage` cyclic KV | BF16 cyclic KV（`W=1024`） |

Inkling2 的 Context 完全驻留 Device 内存（`host_capacity` 默认且必须为 0），KV 页与定长 `StateImage` 共享每卡 Context 字节池，恢复、前缀复用与抢占遵循[Context 管理](context-management.md)。

## 权重表示、MoE 与离线校准

每层 MoE 包含 128 个路由专家，top-8 sigmoid 路由（`route_scale=8.0`，`use_gate_bias=true`，无共享专家）与软截断 SwiGLU（`gate_cap=4.0`，`up_cap=25.0`）。词表权重零填充至 64 行对齐（`vocab_size=200064`，`unpadded_vocab_size=200058`）。转换映射见 [inkling2 converter](../../tools/convert/inkling2.py) 与 [official_recipes](../../tools/convert/official_recipes.py)：

| 角色 | 物理表示与布局 | Activation 许可 |
|---|---|---|
| 路由专家（layers 0–19，40 tensors） | 原生 `nvfp4_mul`（`batched_block_scale_k16_m128x4_v1`，deinterleaved `gate+up` 与 `down` 两 parent） | 基础转换 `A16Only`；发布产物附带离线 `max` 校准乘子（`per_expert_down`，88 个 FP32 张量），启用 `AllowA4`（132 uses） |
| 路由专家（layers 20–23，8 tensors） | 原生 `mxfp8`（`row_group_scale_v1`） | `AllowA8`（12 uses） |
| 词表、注意力/短卷积/输出投影、循环控制器、Vision、DSpark | BF16（`contiguous_le_v1`） | `A16Only` |
| Norm、Router/Bias、短卷积核、相对偏置表、AttnRes Query、Clip scale、校准乘子 | FP32 | 保持原精度 |

原生 `nvfp4_mul` 与 `mxfp8` 码字及 scale 直接保值导入，不反量化再量化。每个逻辑访问（不同 loop pass）拥有独立的 `Use` 记录，使同一物理循环层的不同 pass 可绑定各自的激活校准乘子。

## DSpark 与 Markov RNN Head

DSpark 为 3 层密集草稿模型（`hidden=2304`，`intermediate=9216`，24 Q / 8 KV heads，`head_dim=128`，RoPE `theta=1e6`，`block_size=8` 支持 `1..7` drafts、默认 `7`）：layer 0/2 使用 `W=1024` cyclic SWA，layer 1 使用 full paged KV。

- **Conditioning**：从主干 4 个逻辑位点（`target_layer_ids=[5, 23, 41, 48]`）截取隐藏态，经各层 `fuse_alpha` 与 `dflare_norm`（`weighted_sum_rmsnorm`）融合后投影出 `context_kv` 并应用 `rmsnorm_rope` 写入草稿上下文。
- **Masked-Block Proposal + Markov RNN**：以 anchor token 与 `mask_embed` 组成非因果查询块，通过 `block_sliding_window_attention` / `context_softmax_attention` 产出草稿隐藏态，再由递推 Markov RNN head（`markov_rank=256`，`markov_head_type=rnn_hidden`）结合跨轮持久状态 `dspark_f_rnn` 顺序产生候选 token 与条件概率。其 unembed/hidden 投影在转换期以 FP64 离线预折叠（`dspark_fold=offline_fp64_unembed_hidden_out`）。
- **权重共享与头选择**：DSpark 共享 Text 的 `token_embedding`、`embed_norm` 与 `output_head`，仅支持 `--proposal-head full`。接受/提交语义与自适应验证长度见[Speculative 执行](speculative-decoding.md)，实现见 [execution/dspark](../../src/models/inkling2/execution/dspark.cpp) 与 [program/dspark](../../src/models/inkling2/program/dspark.cpp)。

## PE-Lang Vision 与 Frontend

Vision 为 23 层 PE-Lang ViT（`hidden=1024`，16 heads，`head_dim=64`，`intermediate=4096`，20×20 原生 patch，2×2 空间合并即 40×40 px/token），每张图像带 1 个 CLS token，结合 32×32 双线性插值位置编码与 2D 交错 RoPE（`rope_2d_interleaved`），并由 2 层 MLP projector 映射至 `hidden=2304`。处理器仅支持图像输入（显式拒绝视频），按 CLIP→BF16→PE `[-1, 1]`→BF16 查表归一化，复用共享媒体缓存与 `--vision-memory-budget` 有界窗口。

Frontend 使用 `o200k` BPE 分词器与块级协议标签（`<|message_model|>`、`<|content_thinking|>`、`<|content_text|>`、`<|content_invoke_tool_json|>`、`<|end_message|>`），支持 `none..xhigh/max` 与连续数值 `[0.0, 0.99]` 的 `reasoning_effort`（默认 `high=0.9`）、按需闭合 thinking 的控制后缀（普通正文为 `<|end_message|><|message_model|><|content_text|>`，必选工具调用时为 `<|end_message|><|message_model|>`），并在关闭 thinking 时通过直接正文起始或工具 grammar 禁止生成思考块。详见 [frontend](../../src/models/inkling2/frontend/) 与[Frontend 架构](frontend.md)。

## 执行路径与 CUDA Graph

当前 Inkling2 为单卡 RTX 5090（`sm_120a`）驻留执行，默认 `prefill_chunk=4096`。Prefill、Replay 与 Vision encoder 使用 eager 路径（纯文本多请求 Prefill 支持原生 ragged 打包）；Decode 通过 `DecodeGraphSuite` 捕获主干 target、DSpark proposal、设备侧 acceptance/sampling decision 及 speculative commit，并按 `ContextBucket` 动态修补注意力 kernel 参数。启用自适应 DSpark 时，narrow target 图目录按容量门控覆盖 `{2, 4, K+1}` 验证宽度，避免低接受率场景始终付出全宽验证成本。实现见 [program/graphs](../../src/models/inkling2/program/graphs.cpp) 与 [startup](../../src/models/inkling2/startup.cpp)。
