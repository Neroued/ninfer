# Inkling-2 整模 RTX 5090 接入与组件实施方案

> [!NOTE]
> **范围声明**：本文档定义 Inkling-2 架构（以 `thinkingmachines/Inkling-2-Mini` 为首个实盘验证实例，同时覆盖非循环与循环变体）在 NInfer 公共 Engine 中的完整整模接入方案，涵盖 `.ninfer` V3 转换与权重表示合同、Frontend / Tokenizer / OutputSession 边界、独立 SWA KV 精度控制、Main Loop Decoder + CUDA Graph、23 层 PE-Lang Vision Encoder、3 层 DSpark + Markov RNN 投机后端，以及后续在独立校准分支完成 `A16Only → AllowA4` 离线激活校准的对接协议。
> 算子接口与 RTX 5090 Roofline 规范由[扩展 Op 规范](inkling2-extended-ops-rtx5090.md)与[新增 Op 规范](inkling2-new-ops-rtx5090.md)维护，本文不重复声明算子签名。

---

## 1. 全局架构原则与已确认决策

1. **统一架构入口（单模型家族，支持非循环与循环变体）**：
   - 目录与命名空间：`src/models/inkling2/`（`ninfer::models::inkling2`），注册为 `Architecture::Inkling2`；
   - `.ninfer` artifact 中 `text` 组件统一记录 `architectures = ["Inkling2ForCausalLM"]`、`model_type = "inkling2_text"`，由可选 `loop_config` 决定是否执行 `prelude → core × num_passes → coda`；
   - 首个交付目标为 **RTX 5090 单卡**（`LocalExecutor`），但保持 `ModelDefinition / ModelShard / Session / RankProgram` 所有权分层，不将单卡假设渗入配置、绑定或前端合同。
2. **先完整功能（`A16Only` NVFP4_MUL + `AllowA8` MXFP8），后在独立分支完成离线校准**：
   - `feat/inkling2` 先基于原生权重直接跑通并验收完整功能（Text Loop、Thinking/Tools/Grammar、Vision、DSpark、BF16/FP8 KV 与 SWA KV、CUDA Graph、Prefix Cache、Packed Prefill）；
   - 离线校准（在独立分支挂接 `ActivationCapture` 生成 `AllowA4` 产物）与主功能分支通过 `.ninfer` V3 `Use` 合同严格解耦：主分支的加载与执行层始终按 `Use` 权限透明分发，后续换入已校准 `AllowA4` 产物时零代码修改。
3. **权重与词表精确合同**：
   - **源权重保值导入**：L0–19 MoE 保持原生 `nvfp4_mul`（`multiplier = amax[e] / outer_scale_denominator`），L20–23 MoE 保持原生 `mxfp8`（运行时动态块量化，无需离线激活校准）；
   - **`w13` 行序去交错**：源 `w13_weight` 为交错行 `(gate_0, up_0, gate_1, up_1, ...)`；转换器对 `w13_weight` 与 `w13_weight_scale` 同步按偶数/奇数行重排为 `[gate(0..I-1); up(0..I-1)]`，100% 逐位保值、零重量化；
   - **Per-visit MoE `Use` 与校准粒度预留**：24 物理层权重只存一份；Loop Core（L6–17）按逻辑访问位点（`pass 0/1/2`）声明独立 `Use`（48 个逻辑层访问中 44 个 NVFP4 visit + 4 个 MXFP8 visit）。后续校准分支按 **`w13` 每 visit 标量 `[1]`**（命中 `softcap.cu` 8× 更轻的 broadcast 输入量化）与 **`w2` 每 visit × per-expert `[128]`**（贴合 SoftcappedSwiGLU 后各 expert 动态范围）回填 `activation_input_multiplier`；
   - **词表尾部无效 padding 裁齐**：实盘 `unpadded_vocab_size = 200058`、`vocab_size = 201024`；转换器将 `embed`、`unembed`、`markov_w1` 截齐至 `200064` 行（`ceil(200058 / 64) * 64`，保留全部 `200058` 有效行 + 6 行零填充），直接命中已调优的 `n200064_k2304`、`markov_head` 与 `speculative_accept` 内核；
   - **DSpark `fold_weight / fold_bias` 离线 FP64 预折叠**：转换器用 FP64 精确计算 `fold_weight = unembed[:200064] @ hidden_out.weight`（`[200064, 256]` BF16，200058..200063 置零）与 `fold_bias = unembed[:200064] @ hidden_out.bias`（`[200064]` BF16）写入 `draft` 组件，启动零计算开销。
4. **独立 SWA KV 精度控制（对标 `linear_state_dtype`）**：
   - 实盘 48 逻辑层为 **40 SWA（window=1024）+ 8 Full（page=64）** + `48 × 4` 条 ShortConv width-3 状态：
     - 8 层 Full PagedKV 随上下文长度线性增长（32K 下 BF16 1.0 GiB / FP8 516 MiB），由现有 **`engine.kv_dtype` / `--kv-dtype`**（`bf16|fp8`）控制；
     - 40 层 SWA CyclicKV 为定长 State（单序列 BF16 **160.0 MiB** / FP8 **80.6 MiB**，与 1.83 MiB ShortConv 同驻 `StateImageStorage`），在存储位置、定长体积、checkpoint 语义和精度敏感度上与 `linear_state` 完全同构；
   - 在 `EngineOptions` 与 [schema.cpp](../../src/product/config/schema.cpp) 新增独立配置项 **`engine.swa_kv_dtype` / `--swa-kv-dtype`**，支持 `bf16|fp8`，**默认 `bf16`**。支持 `Full=FP8 + SWA=BF16`（推荐长上下文高精度组合）、`Full=FP8 + SWA=FP8`（极限并发）、`Full=BF16 + SWA=BF16`（全精度基线）；
   - DSpark 的 3 层 Context KV（2×SWA 共 8 MiB/seq + 1×Full）固定为 BF16/FP16；`speculative.backend` / `--spec` 新增 **`dspark`**。

---

## 2. 实例几何核对（`thinkingmachines/Inkling-2-Mini`）

本地源权重路径：`/home/neroued/models/llm/inkling/Inkling-2-Mini/base-hf/`（commit `c9d028302cd0`）。

| 子系统 | 关键几何参数 |
|---|---|
| **Main Text** | `H=2304`, `num_hidden_layers=24`, `loop_config={core_start:6, core_end:18, num_passes:3, gate_rank:64}` → **48 逻辑层**（6 prelude + 12×3 core + 6 coda）；`unpadded_vocab_size=200058`, `padded_vocab=200064`; `Hq=24, Hkv=8, D=128, d_rel=16, rel_extent=1024, swa_window=1024`; Full 物理层 `{5,11,17,23}` → 8 Full 逻辑访问 + 40 SWA 逻辑访问；`n_layers_per_attn_res_block=6`, `sconv_kernel_size=4`, `log_scaling=(8192, 0.1)`, `qk_clip=16√2`, `rel_clip=8.0`, `final_logit_softcapping=30.0` |
| **Main MoE** | `E=128, K=8, I=1152, n_shared_experts=0`, `route_scale=8.0`, `SoftcappedSwiGLU(4.0, 25.0)`；L0–19 为 `NVFP4_MUL`（44 逻辑 visit），L20–23 为 `MXFP8`（4 逻辑 visit） |
| **Vision** | `23` 层 PE-Lang ViT，`H=1024, heads=16, D=64`, `native_patch_size=20, adapter_merge_size=2`（40×40 pixel cell）, `posemb_grid_size=32`, `mlp_ratio=4.0`, `ln_eps=1e-5`, projector `4096 → 4096 (GELU) → 2304`（无 bias） |
| **DSpark (`mtp.safetensors`)** | `3` 层（`local_layer_ids=[0, 2]` SWA 1024，L1 Full），`H=2304, Hq=24, Hkv=8, D=128, I=9216`, `rope_theta=1e6`, `block_size=8`（1 anchor + 7 mask）, `markov_rank=256`, `target_layer_ids=[5, 23, 41, 48]`（对应第 5/23/41 逻辑层输入 partial block 及 final-norm 输出 `48`）；共享 Main `embed` / `embed_norm` / `unembed` |

---

## 3. 组件合同与目录规划

### 3.1 C1 · Converter & `.ninfer` V3 Schema（`tools/convert/inkling2.py`、`official_recipes.py`）

- **组件划分**：`text`（必需）、`vision`（可选）、`draft`（可选，来源 `mtp.safetensors`）。
- **关键参数映射与转换规则**：
  - **Main Attention 合并投影**：
    - SWA 物理层：`q_r_gate_k_v` 合并为连续 parent `[8576, 2304]`（顺序：`Q 3072 | R 384 | gate 3072 | K 1024 | V 1024`），在 pass 0 一次 `linear_split(8576, 2304)` 输出 5 路；在 Loop Core pass > 0 时，前 `6528` 行借用为 `q_r_gate`（`linear_split(6528, 2304)`），后 `2048` 行拆分为 `K`/`V` 逻辑角色供 `linear_pair(1024, 2304)`；
    - Full 物理层：`q_r_gate` 合并为 `[6528, 2304]`，`K`/`V` 独立为 `[1024, 2304]`，`k_sconv.weight_proj` / `v_sconv.weight_proj` 为 `[32, 2304]`；
    - `attn_sink_proj.weight` 保留为 `[24, 2304]`（由 `relative_attention_prepare` 融合计算，零独立 GEMM）；
    - `rel_logits_proj.proj`：从 checkpoint `[16, 1024]` 转置为 `[1024, 16]`（NInfer `{16, 1024}`，`d_rel=16` fastest）；
    - ShortConv 权重：SWA `k_sconv.weight`/`v_sconv.weight` `[1024, 1, 4]` 转置为 `[4, 1024]`（NInfer `{1024, 4}`，oldest-to-newest）；Full `k_sconv.static_w`/`v_sconv.static_w` `[4, 1024]` 原样保留（newest-to-oldest）；`attn_sconv.weight`/`mlp_sconv.weight` `[2304, 1, 4]` 转置为 `[4, 2304]`（NInfer `{2304, 4}`）；
  - **Block-AttnRes & Loop Controllers**：
    - `attn_res.attn_queries.{0..22}` 绑定至物理层 `1..23`（层 0 无 `attn_query`），保持 FP32 原始权重，执行期传入 `query_scale = 1/sqrt(H)`；
    - `middle_cycle_cross_pass_queries`（`[24, 2304]`）与 `middle_cycle_cross_pass_kv_queries`（`[24, 2304]`）按每个 core 层重组为 `[4, 2304]` FP32（顺序 `attn, k, v, mlp`）；
    - Core 层 `controllers.{attn,k,v,mlp}` 按 `LoopMessageWeights` 合同堆叠为 `attn_kv_down [3, 64, 4608]`、`attn_kv_up [3, 4608, 64]`（`[gain; forget]`）以及 `mlp_down [1, 64, 4608]`、`mlp_up [1, 4608, 64]`；
  - **Main MoE**：
    - L0–19：新增 `InklingNvfp4Source` 读取 `w13/w2_weight`（U8）、`*_scale`（F8_E4M3）、`*_amax`（F32 `[128]`）、`*_outer_scale_denominator`（F32 `[1]`），计算 per-expert `multiplier = amax[e] / denom`，对 `w13` 执行偶/奇行去交错为 `[gate(0..1151); up(0..1151)]` 后直接经 `import_encoded` 写入 `nvfp4_mul`（`batched_block_scale_k16_m128x4_v1`）；
    - L20–23：新增 `InklingMxfp8Source` 读取 `w13/w2_weight`（F8_E4M3）与 `*_scale`（U8 UE8M0），对 `w13` 同步偶/奇行去交错后直接经 `import_encoded` 写入 `mxfp8`（`AllowA8`）；
    - Per-visit `Use`：`text/layers/{l}/moe/experts/{gate,up,down}` 对每个逻辑 pass `p` 声明独立 input 名 `mlp_in_p{p}` / `mlp_act_p{p}`；
  - **Vision**：`q_proj/k_proj/v_proj` 权重与 bias 合并为 `qkv_proj [3072, 1024]` + `qkv_bias [3072]`；其余 1:1 映射；
  - **DSpark (`draft`)**：
    - `q/k/v_proj` 合并为 `[5120, 2304]`，`k/v_proj_ctx` 合并为 `[2048, 2304]`，`gate/up_proj` 合并为 `[18432, 2304]`；
    - `markov_head.rnn_proj.weight`（`[768, 5120]`）沿列拆为 `w_recurrent [768, 512]`、`w_hidden [768, 2304]`、`w_frnn [768, 2304]`，`rnn_proj.bias` 转 FP32 `[768]`；
    - `markov_w1` 裁齐至 `[200064, 256]`；`fuse_weight` 转 `fuse_alpha = softmax(fuse_weight.float(), dim=-1)` FP32 `[4, 4]`；离线 FP64 计算 `fold_weight [200064, 256]` 与 `fold_bias [200064]`；
    - 共享 `text/token_embedding`、`text/embed_norm`、`text/output_head`。
  - **Recipe**：`inkling2`（默认导出 `A16Only` NVFP4_MUL + `AllowA8` MXFP8，同时支持通过 `--calibrate` 写入 `activation_calibration` 计划供后续校准分支消费）。

### 3.2 C2 · 公共文本扩展 & Inkling Frontend 边界

| 层级 | 文件 | 职责与边界 |
|---|---|---|
| 公共 `src/text/` | `src/text/tokenizer.cpp` | 新增 `BpeSplit::O200k` 预分词规则（匹配 Inkling `tokenizer.json` 的 `o200k_base` 正则），其余 `ByteLevel` / BPE / added special tokens 完全复用 |
| 公共 `src/models/common/frontend/` | `output_session.{h,cpp}` | 在不改动现有 Qwen/GLM `ReasoningFormat` 路径的前提下，增加“全权块解码”模式：当 `BodyDecoder` 接管结构化输出时，向其传递含结构控制 token 的流，允许其直接产出 `Reasoning / Content / ToolCall` 事件并提供 `thinking_budget` 强制闭合钩子 |
| Inkling 专属 `src/models/inkling2/frontend/` | `resources.{h,cpp}`, `chat_template.{h,cpp}`, `prompt_layout.{h,cpp}`, `processor.{h,cpp}`, `block_decoder.{h,cpp}`, `tool_contract.{h,cpp}`, `tool_grammar.{h,cpp}`, `output.{h,cpp}`, `frontend.{h,cpp}` | ① 参数规范化（7 档 `ReasoningEffort` + `[0.0, 0.99]` 浮点 + `enable_thinking=false → 0.0` + `developer → system`）；② 复用 `third_party/llama-jinja` 渲染 `chat_template.jinja`（保留 `literal_spans` / `regions`）并追加生成前缀（`<|message_model|>` 或续写头）；③ 块级状态机解析 `<|content_thinking|>` / `<|content_text|>` / `NAME<|content_invoke_tool_json|>` / `<|end_message|>` 与 `<|content_model_end_sampling|>` 停止；④ XGrammar 约束（JSON Schema、Tool Choice `auto|required|none`、`parallel_tool_calls`）；⑤ 图像预处理（alpha 棋盘格合成 → 40×40 patch → `-1` padding → 展开 `<|unused_200054|>`，拒绝音视频，支持 `user` 与 `tool` 角色图片） |

### 3.3 C3 · 配置、模型定义、权重加载与新配置项

- **全局配置项扩展**：
  - `include/ninfer/types.h`、`src/product/config/{schema,options,configuration}.cpp`、`src/runtime/engine/engine.cpp`、`include/ninfer/observability/snapshot.h`：
    - 新增 `KvCacheStorage swa_kv_cache = KvCacheStorage::BFloat16`（配置键 `engine.swa_kv_dtype`，CLI `--swa-kv-dtype bf16|fp8`，默认 `bf16`）；
    - `SpeculativeBackend` 新增 `DSpark`（配置值 `dspark`，`draft_tokens` 默认 `7`，校验 `1..block_size-1`）。
- **模型定义与加载**（`src/models/inkling2/{config,definition,weights,load,load/*}`）：
  - `Config` 解析 `text`（含可选 `loop_config`）、可选 `vision`、可选 `draft`（`mtp_config`）及派生逻辑层表 `logical_layers[48]`（记录 `physical_layer`, `pass_index`, `is_swa`, `swa_index` / `full_index`, `block_index`, `starts_new_block`, `capture_aux_slot`）；
  - `ModelDefinition` / `ModelShard` 复用 `common/bindings.cpp` 与 `common/weights.cpp`，加载期读取每个 visit 的 `WeightUse`（自动兼容 `A16Only` 与后续已校准 `AllowA4` + `activation_input_multiplier`）。

### 3.4 C4 + C5 · State/Context/Session、Main Loop Decoder 与 CUDA Graph

- **统一 Context 存储布局**（`src/models/inkling2/state/`）：
  - **定长 `StateImageStorage`**：
    - `40` 层 Main SWA `CyclicKVCacheLayout`（`capacity=1024, Hkv=8, D=128`，精度由 `engine.swa_kv_dtype` 控制：`BF16` 160 MiB / `FP8` 80.6 MiB）；
    - `48` 层 ShortConv 状态（每层 4 条 width-3 BF16 状态：`k_conv [1024,3]`, `v_conv [1024,3]`, `attn_sconv [2304,3]`, `mlp_sconv [2304,3]`，合计 1.83 MiB）；
    - （可选 DSpark）`2` 层 DSpark SWA `CyclicKVCacheLayout`（`capacity=1024`，BF16/FP16，8 MiB）+ `f_rnn` / anchor aux 缓存；
  - **动态 `PagedKV` 页池**：
    - `8` 层 Main Full `PagedKV`（`page=64, Hkv=8, D=128`，精度由 `engine.kv_dtype` 控制：`BF16` 或 `FP8`）；
    - （可选 DSpark）`1` 层 DSpark Full `PagedKV`（BF16/FP16）。
- **小算子补齐（S4 前置，1 个极小形状）**：
  - Full 物理层的 `k_sconv.weight_proj` 与 `v_sconv.weight_proj`（各 `[32, 2304]`）新增 `linear_pair` / `linear_projection` 的 `n32_k2304` RTX 5090 BF16 形状（1 次发射输出两路连续 `[32, T]`，权重仅 288 KB，避免改变现有 `linear_pair n1024_k2304` 合同）。
- **Main Loop Decoder 执行流**（`src/models/inkling2/execution/text.cpp`）：
  1. `embedding(tokens)` → `rmsnorm(embed_norm)`；若有 Vision 特征，**在 `embed_norm` 之后**将 Vision embeddings 写入 `<|unused_200054|>` 位置；
  2. 初始化 Block-AttnRes 残差栈 `stack[max_blocks]`（对 `core=[6,18), block=6` 最多 4 个已完成块 + 1 个 running partial）；
  3. **Prelude（L0–5）** → 保存 `prelude_partial` 与 `prelude_blocks` → **Core（L6–17 × 3 passes）**：
     - `pass == 0`：普通层执行；
     - `pass > 0`：每轮从上一轮 `bank`（上一轮 core 产出的 2 个块）经 `block_attn_res(queries=message_queries[4])` 生成 4 路 `messages[attn,k,v,mlp]`，再经 `loop_message_blend_rmsnorm(S=3)` 融合 Q/K/V 输入、`loop_message_blend_rmsnorm(S=1)` 融合 MLP 输入；
  4. **Coda（L18–23）** → 最终 `block_attn_res(output_query, norm_weight=final_norm)` → `linear(unembed, FP32)` → `logit_softcap(30.0)`；
  5. **每层内计算**：
     - 投影：SWA pass 0 走 `linear_split(8576, 2304)`；Full 与 SWA pass > 0 走 `linear_split(6528, 2304)` + `linear_pair(1024, 2304)`（Full 额外 `linear_pair(32, 2304)` 生成 `k_dyn, v_dyn`）；
     - `relative_attention_prepare`（融合 Q/R clip+log-scale、K/V static/dynamic conv、K clip / V RMSNorm、BF16/FP8 编码写入 PagedKV 或 SWA `chunk_kv` / W=1 直写 CyclicKV、以及 `sink` 投影）；
     - `relative_softmax_attention`（SWA/Full、Decode/Verify/Ragged、BF16/FP8）→ SWA 非 W=1 路径在注意力和校验完成后经 `relative_attention_commit_cyclic_kv` 提交已接受前缀；
     - `linear(wo_ud 2304×3072)` → `short_conv_block_attn_res`（融合 `attn_sconv` + MLP 前 AttnRes）；
     - `topk_sigmoid_route` → `routed_swiglu`（L0–19 `NVFP4_MUL` / L20–23 `MXFP8`，`SoftcappedSwiGLU(4.0, 25.0)`）→ `short_conv_block_attn_res`（融合 `mlp_sconv` + 残差更新 + 下一层 AttnRes）；
     - 若开启 DSpark，在 `logical_layer ∈ {5, 23, 41}` 的输入 partial 及最终 `final_norm` 输出（slot `48`）写入 `aux_hidden[4, H, T]`。
- **CUDA Graph & 执行模式**：
  - Prefill（含纯文本多请求 Packed Prefill）与 Replay 走 Eager + Ragged 算子；
  - Decode 与 Speculative Verify 走扁平 `DecodeGraphSuite` + `ContextBucketSite`（利用已修复的 `DecodeGraphKernelPatch` PDL 边支持），精确覆盖 Full KV bucket 切换、ShortConv snapshot/commit 与 SWA `chunk_kv` commit。

### 3.5 C6 · Vision Encoder（`src/models/inkling2/vision/`）

- 接入 `common/vision/{resources,window,execution}` 有界媒体窗口预算；
- 图像执行流（每图独立 attention segment）：
  1. CPU 预处理输出 `[N_cell, 40, 40, 3]` 经精确 CLIP → PE `[-1, 1]` 归一化并重排为原生 `[N_native, 1200]`（20×20×3）patch；
  2. `linear(1024×1200)` → 插入每图 CLS 行 + `embedding_interpolate_add`（32×32 双线性插值位置编码，CLS 行叠加 `class_embedding + cls_position_embedding`）→ `layer_norm(ln_pre)` → `layer_norm(L0.ln_1)`；
  3. 23 层 ViT：`linear_split(3072×1024+bias)` → `rope_2d_interleaved`（CLS 行恒等）→ `packed_softmax_attention(D=64, H=16)` → `linear_bias(1024×1024)` → `scaled_residual_add_layer_norm(ls_1, ln_2)` → `linear_bias(4096×1024)` → `gelu` → `linear_bias(1024×4096)` → `scaled_residual_add_layer_norm(ls_2, next_ln_1)`；
  4. 去除 CLS 行按 2×2 merge 次序重排 `[N_cell, 4096]` → `linear(4096×4096)` → `gelu` → `linear(2304×4096)` → 直接写入 Main `embed_norm` 之后的对应 `<|unused_200054|>` 位置。

### 3.6 C7 · DSpark 3 层投机后端 + Markov RNN Head（`src/models/inkling2/{execution/dspark,program/dspark}`）

- **Context 物化**（Prefill 尾部及每轮 Accept 后）：
  - 对已提交 token 的 `aux_hidden[4, H, T]`，每层 `l ∈ {0, 1, 2}` 执行 `weighted_sum_rmsnorm(fuse_alpha[l], dflare_norm)` → `linear_split(2048×2304)` → `rmsnorm_rope(k_norm_ctx)` → `kv_cache_append`（L0/L2 写 1024 槽 CyclicKV，L1 写 PagedKV）；
  - 对每条请求最后一个已提交 token，用 `fuse_alpha[3]` + `markov_head.dflare_norm` 计算 `f_rnn [H, B]`。
- **Draft Block 推理与自回归 Markov 采样**：
  - 构造 `[anchor, mask × 7]`（anchor 走共享 `embed → embed_norm`，mask 位置填充 `mask_embed`）；
  - 3 层 Draft：`rmsnorm` → `linear_split(5120×2304)` → `rmsnorm_rope(24, 8)` → `sliding_window_attention`（L0/L2）或 `context_softmax_attention`（L1）→ `linear_add(2304×3072)` → `rmsnorm` → `linear_swiglu(18432×2304)` → `linear_add(2304×9216)` → `rmsnorm_pack_tail(H=2304)` 取出 7 个 draft 位置；
  - `linear(unembed 200064×2304, BF16)` 得 `base_logits`；`markov_rnn_context(hidden, f_rnn)` 一次预计算 7 步上下文；
  - 串行 7 步：`markov_rnn_step` → `markov_logits_add_softcap(token_domain=200058, softcap=30.0)` → 贪心/采样得 `draft_token[i]` 并作为下一步 `prev_tokens`；
- **Target 验证与接受**：
  - Target 一次并行 Verify `W = 1 + draft_tokens` 个 token（ShortConv snapshot + SWA `chunk_kv`），经 `speculative_accept_sparse_drafts`（`token_domain=200058, physical_rows=200064`）判定接受长度 `accepted`，随后提交 ShortConv snapshot 与 SWA `relative_attention_commit_cyclic_kv` 并增量物化 DSpark Context KV。

---

## 4. 实施阶段、并行策略与验收门禁

```text
S1: Converter & Recipe + 实盘转换
 │
 ├──▶ S2: Tokenizer(O200k) + OutputSession 块解码扩展 + Inkling Frontend [与 S3 并行]
 ├──▶ S3: Config / Definition / Loading + engine.swa_kv_dtype & dspark 配置 [与 S2 并行]
 │
 └──▶ S4: n32_k2304 + State/Context/Session + Main Loop Decoder + CUDA Graph + Engine 注册
       │  ★ Milestone 1：实盘 Text 端到端数值对齐参考 + BF16/FP8 Full & SWA KV PPL 验证
       │
       ├──▶ S5: Vision Encoder (23 层 PE-Lang ViT) [与 S6 并行]
       └──▶ S6: DSpark 3 层投机后端 + Markov RNN Head [与 S5 并行]
             │
             ★ Milestone 2：整模完整功能收口（Text + Thinking/Tools/Grammar + Vision + DSpark）
             │
             └──▶ [后续独立校准分支] ActivationCapture + finalize_calibration → AllowA4 产物与 PPL/吞吐验收
```

| 阶段 | 提交粒度 | 强制验收标准 |
|---|---|---|
| **S1** | `feat(convert): add Inkling2 checkpoint adapter and official recipe` | Python 单测验证 `nvfp4_mul`/`mxfp8`/`w13` 去交错/词表裁齐/FP64 `fold_weight` 精确性；实盘 `Inkling-2-Mini` 转换成功并通过 `Artifact` 校验 |
| **S2** | `feat(text): add o200k BPE pre-tokenizer split mode` + `feat(inkling2): add Inkling frontend, block output decoder, and image processor` | Tokenizer 对照实盘 HF `tokenizer.json` 逐 token 一致；Chat 渲染对照参考一致；流式 thinking/content/tool_calls + XGrammar + ImageProcessor CTest 全过 |
| **S3** | `feat(config): add swa_kv_dtype and dspark speculative backend options` + `feat(inkling2): add config, model definition, and weight loading` | 实盘 `.ninfer` 加载校验单测通过；跨目标编译无警告 |
| **S4** | `feat(linear): add RTX 5090 BF16 32x2304 pair projection` + `feat(inkling2): add state, main loop decoder, CUDA graph, and engine registration` | 实盘 `Inkling-2-Mini`（`A16Only`）首步与多步 logits 对照独立 PyTorch 参考验证；Eager vs CUDA Graph 一致；`Full(bf16/fp8) × SWA(bf16/fp8)` PPL 验证；CLI / Serve / Prefix Cache / Packed Prefill 通过 |
| **S5** | `feat(inkling2): add PE-Lang vision encoder and multimodal execution` | Vision feature 对照 PyTorch 参考核验；单图/多图/透明图/非整除尺寸端到端生成与媒体缓存测试通过 |
| **S6** | `feat(inkling2): add DSpark speculative backend and Markov RNN head` | DSpark 贪心/采样/约束解码与非投机输出一致；实盘接受率与 RTX 5090 吞吐实测报告；多轮/Prefix Cache/中断恢复回归全过 |
