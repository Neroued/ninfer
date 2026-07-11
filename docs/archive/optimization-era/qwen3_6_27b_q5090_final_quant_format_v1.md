# Qwen3.6-27B 最终权重量化存储规格：`q5090_w4g64_mixed_v1`

版本：v1.0-final  
目标 runtime：单机 / 单 RTX 5090 32GB / 单用户 C++/CUDA 推理  
目标模型：`Qwen/Qwen3.6-27B` text core + optional MTP + optional vision/mmproj，兼容 Qwen3.5/Qwen3.6 27B 同类配置  
固定内部格式：

```text
TEXT_CORE:       q5090_w4g64_mixed_v1
MTP_DRAFT:      mtp_w8g128_v1
VISION_ENCODER: vision_q4mix_merger_w8g128_v1
fallback:       vision_merger_bf16_strict
```

主目标：接近 GGUF `Q4_K_M` 的 text 质量，但使用更简单、更 CUDA 友好的固定内部 layout；MTP 与 vision/mmproj 作为独立 optional segment 管理，后期可加入而不破坏 text core ABI。

> 最终约定：`TEXT_CORE` 必须可单独加载并完成 text-only 推理；`MTP_DRAFT` 和 `VISION_ENCODER` 是 optional module segment。converter 必须支持 `--include-mtp` 与 `--include-vision`，runtime 必须允许 segment resident / lazy-load / free。

---

## 0. 配置锁定

converter 必须读取 `config.json` 并 assert 下表；不匹配时不要生成本格式。

| 字段 | 值 | 实现含义 |
|---|---:|---|
| `num_hidden_layers` | 64 | text layer 编号 0..63 |
| `full_attention_interval` | 4 | 每 4 层一个 full attention |
| full attention layers | `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63` | 16 层 |
| linear/GDN layers | `0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, 44-46, 48-50, 52-54, 56-58, 60-62` | 48 层 |
| `hidden_size` | 5120 | 所有主线性层输入维度 |
| `intermediate_size` | 17408 | FFN gate/up 输出、down 输入 |
| `vocab_size` | 248320 | embedding 与 lm_head 行数 |
| full attention `num_attention_heads` | 24 | query/output channel = 24 × 256 = 6144 |
| full attention `num_key_value_heads` | 4 | KV channel = 4 × 256 = 1024 |
| full attention `head_dim` | 256 | q/k norm 长度 |
| GDN `linear_num_key_heads` | 16 | GDN q/k channel = 16 × 128 = 2048 |
| GDN `linear_num_value_heads` | 48 | GDN v/z/out channel = 48 × 128 = 6144 |
| GDN `linear_conv_kernel_dim` | 4 | `linear_attn.conv1d.weight` kernel |
| `max_position_embeddings` | 262144 | 长上下文上限；常规 KV cache 只来自 16 个 full attention 层 |

---

## 1. 量化 primitive

### 1.1 qtype 枚举

| qtype | 用途 | bits/weight，含 scale | group | scale | zero/min | signed range |
|---|---|---:|---:|---|---|---|
| `Q4G64_F16S` | 默认大矩阵 | 4.25 bpw | 64 K-elements | FP16 per row per K64 group | 无 | [-8, 7] |
| `Q5G64_F16S` | 敏感投影 | 5.25 bpw | 64 K-elements | FP16 per row per K64 group | 无 | [-16, 15] |
| `Q6G64_F16S` | embedding / lm_head | 6.25 bpw | 64 K-elements | FP16 per row per K64 group | 无 | [-32, 31] |
| `W8G128_F16S` | MTP / vision merger | 8.125 bpw | 128 K-elements | FP16 per row per K128 group | 无 | [-128, 127]，实际 clamp 到 [-127,127] |
| `BF16_CTRL` | norm、小 gating、小 conv、非 GEMM 控制权重 | 16.00 bpw | 不分组 | 无 | 无 | BF16 |
| `FP32_CTRL` | GDN decay / bias 等指数敏感标量 | 32.00 bpw | 不分组 | 无 | 无 | FP32 |

所有 `QxG64_F16S` 与 `W8G128_F16S` 都是 signed symmetric weight-only quantization：

```text
group = W[row, k0:k0+group_size]
qmax  = 2^(bits-1) - 1
scale = choose_scale(group, qmax)
q_i   = clamp(round(w_i / scale), -qmax-1, qmax)
w_i≈  scale * q_i
```

默认 `choose_scale = max(abs(group)) / qmax`。正式模型建议使用 calibration activation RMS 做 weighted-MSE scale search，但文件格式不依赖 calibration 算法。

v1 不存 `zero` / `min`。text core 敏感 tensor 通过 Q5/Q6 处理；MTP 与 vision merger 通过 W8 处理，而不是用 asymmetric INT4/INT8 增加 hot path metadata 和 integer subtract。

---

## 2. Runtime layout

### 2.1 大矩阵统一 layout：`TILE_N64_K64`

所有 large linear weight 以逻辑 shape `[N, K]` 存储，即 PyTorch `nn.Linear(in_features=K, out_features=N)` 的 `.weight` shape。文件 layout 与 GPU layout 相同，启动时不做 runtime repacking。

```text
for n_tile in 0 .. ceil(N/64)-1:
  for k_tile in 0 .. ceil(K/64)-1:
    fp16 scale[64]              # 64 rows × one scale for this K64 tile = 128B
    packed_qdata[64 rows][64 K] # row-major bitstream inside the N64×K64 tile
```

| qtype | scale bytes/tile | qdata bytes/tile | tile bytes |
|---|---:|---:|---:|
| `Q4G64_F16S` | 128 | 64 × 32 = 2048 | 2176 |
| `Q5G64_F16S` | 128 | 64 × 40 = 2560 | 2688 |
| `Q6G64_F16S` | 128 | 64 × 48 = 3072 | 3200 |

### 2.1.1 W8 layout：`TILE_N64_K128`

`W8G128_F16S` 专用于 MTP 和 vision merger/mmproj，不与 text core 的 Q4/Q5/Q6 hot path 混在同一个 kernel 分支里。逻辑 shape 仍是 `[N,K]`，但 K tile 固定为 128：

```text
for n_tile in 0 .. ceil(N/64)-1:
  for k_tile in 0 .. ceil(K/128)-1:
    fp16 scale[64]                 # 64 rows × one scale for this K128 tile = 128B
    int8 qdata[64 rows][128 K]     # 8192B, row-major inside tile
```

| qtype | scale bytes/tile | qdata bytes/tile | tile bytes |
|---|---:|---:|---:|
| `W8G128_F16S` | 128 | 64 × 128 = 8192 | 8320 |

`W8G128_F16S` 的 decode kernel 比 Q4/Q5 简单：没有 nibble unpack，没有 5-bit 跨 byte 解码；每个 K128 group 读一次 FP16 scale，int8 转 BF16/FP16 后乘 scale。

### 2.2 Embedding layout：`ROW_GROUPED_G64`

`model.language_model.embed_tokens.weight` 是 row lookup，不走 `TILE_N64_K64`：

```text
for token_id in 0 .. vocab_size-1:
  for k_tile in 0 .. hidden_size/64-1:
    fp16 scale
    packed_qdata[64 K]
```

这样单 token embedding lookup 只读目标 vocab row，不拉取同 tile 内其它 vocab rows。

### 2.3 Alignment

| 对象 | alignment | 说明 |
|---|---:|---|
| file header | 4096B | 便于 mmap/direct read |
| tensor payload offset | 256B，推荐 4096B | GPU coalesced load + 简化 allocator |
| quant matrix shape | N/K pad 到 64 | kernel 无尾块分支 |
| control tensor payload | 64B 或 256B | 小 tensor 可集中放 control arena |

Qwen3.6 的主要矩阵维度均可被 64 整除，padding 主要来自小 control tensor 和文件级对齐。

---

## 3. 全局 tensor 存储

| Tensor | shape | qtype | layout | 说明 | 估算大小 |
|---|---:|---|---|---|---:|
| `model.language_model.embed_tokens.weight` | `[248320,5120]` | `Q6G64_F16S` | `ROW_GROUPED_G64` | token lookup，单独 layout | 0.925 GiB |
| `model.language_model.norm.weight` | `[5120]` | `BF16_CTRL` | contiguous | final RMSNorm | 0.009766 MiB |
| `lm_head.weight` | `[248320,5120]` | `Q6G64_F16S` | `TILE_N64_K64` | logits GEMV/GEMM；保持 Q6 | 0.925 GiB |

公开 index 中同时有 `embed_tokens.weight` 与 `lm_head.weight`。v1 默认分别存储，不假设 tied embedding。若 converter 校验二者 byte-identical，可另做实验选项，但不要改变 v1 基本格式。

---

## 4. 每层量化策略总表

| Layer | 类型 | 实际模块 | 量化摘要 | 层内 packed 大小 |
|---:|---|---|---|---:|
| 0 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 1 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 2 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 3 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 4 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 5 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 6 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 7 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 8 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 9 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 10 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 11 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 12 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 13 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 14 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 15 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 16 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 17 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 18 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 19 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 20 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 21 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 22 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 23 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 24 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 25 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 26 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 27 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 28 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 29 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 30 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 31 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 32 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 33 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 34 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 35 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 36 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 37 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 38 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 39 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 40 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 41 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 42 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 43 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 44 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 45 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 46 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 47 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 48 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 49 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 50 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 51 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 52 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 53 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 54 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 55 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 56 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 57 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 58 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 59 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |
| 60 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 61 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 62 | linear attention / GDN | `linear_attn.*` + `mlp.*` | `q/k=Q4`, `v/z/out=Q5`, `a/b/conv/norm/A_log/dt_bias=BF16/FP32`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 216.817 MiB |
| 63 | full attention | `self_attn.*` + `mlp.*` | `q_proj.q=Q4`, `q_proj.gate=Q5`, `k=Q4`, `v/o=Q5`, `mlp.gate/up=Q4`, `mlp.down=Q5` | 207.364 MiB |

汇总：

| 类别 | 层数 | 单层大小 | 合计 |
|---|---:|---:|---:|
| linear/GDN layer | 48 | 216.817 MiB | 10.163 GiB |
| full attention layer | 16 | 207.364 MiB | 3.240 GiB |
| embedding | 1 | 0.925 GiB | 0.925 GiB |
| final norm | 1 | 0.009766 MiB | 0.009766 MiB |
| lm_head | 1 | 0.925 GiB | 0.925 GiB |
| **text-only total** | - | - | **15.254 GiB / 16.378 GB** |

有效 text-only bits/weight 约 **4.872 bpw**。建议 runtime 为 weight arena 预留 **15.5–16.0 GiB**，覆盖 header、alignment、tensor index、allocator padding 和未来小变体。

---

## 5. Linear attention / GDN 层详细模板

适用 layer：`0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, 44-46, 48-50, 52-54, 56-58, 60-62`。实际 tensor 前缀：

```text
model.language_model.layers.{L}.
```

| Tensor / logical slice | 逻辑/存储 shape | qtype | layout | 说明 |
|---|---:|---|---|---|
| `input_layernorm.weight` | `[5120]` | `BF16_CTRL` | contiguous | RMSNorm；不量化 |
| `linear_attn.A_log` | `[48]` | `FP32_CTRL` | contiguous | GDN decay/transition 标量；保持 FP32 |
| `linear_attn.dt_bias` | `[48]` | `FP32_CTRL` | contiguous | 时间/门控 bias；保持 FP32 |
| `linear_attn.conv1d.weight` | `[10240,4,1]` | `BF16_CTRL` | contiguous | conv 小权重；保持 BF16；M2.8/M3 canonical runtime-native layout |
| `linear_attn.in_proj_a.weight` | `[48,5120]` | `BF16_CTRL` | contiguous | 小 gate/control projection；保持 BF16 |
| `linear_attn.in_proj_b.weight` | `[48,5120]` | `BF16_CTRL` | contiguous | 小 gate/control projection；保持 BF16 |
| `linear_attn.in_proj_qkv.weight::q` | rows `[0:2048]` of `[10240,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | GDN query projection |
| `linear_attn.in_proj_qkv.weight::k` | rows `[2048:4096]` | `Q4G64_F16S` | `TILE_N64_K64` | GDN key projection |
| `linear_attn.in_proj_qkv.weight::v` | rows `[4096:10240]` | `Q5G64_F16S` | `TILE_N64_K64` | GDN value projection，敏感 |
| `linear_attn.in_proj_z.weight` | `[6144,5120]` | `Q5G64_F16S` | `TILE_N64_K64` | GDN output/value gate，敏感 |
| `linear_attn.norm.weight` | `[128]` | `BF16_CTRL` | contiguous | GDN 内部 norm；不量化 |
| `linear_attn.out_proj.weight` | `[5120,6144]` | `Q5G64_F16S` | `TILE_N64_K64` | GDN 输出回 hidden，敏感 |
| `post_attention_layernorm.weight` | `[5120]` | `BF16_CTRL` | contiguous | RMSNorm；不量化 |
| `mlp.gate_proj.weight` | `[17408,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | FFN gate |
| `mlp.up_proj.weight` | `[17408,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | FFN up |
| `mlp.down_proj.weight` | `[5120,17408]` | `Q5G64_F16S` | `TILE_N64_K64` | FFN down，跨中间维聚合，敏感 |

M2.8/M3 canonical TEXT_CORE policy requires `linear_attn.conv1d.weight` to be emitted in the
runtime-native logical shape `[10240,4,1]`. This is a tensor-plan logical-shape policy change, not a q5090
binary container ABI redesign.

Implementation status, 2026-06-27: P1 implements the canonical conv1d sync across converter output,
q5090 fixtures, runtime binding, and tests. Official M2.8/M3 q5090 artifacts must use `[10240,4,1]`.
Existing raw `[10240,1,4]` q5090 files are legacy pre-M2.8 artifacts. They must be regenerated before
they are used as official M2.8/M3 baseline inputs.

This policy is synchronized in:

- `tools/q5090_convert/tensor_plan.py`;
- `tests/fixtures/make_q5090_fixture.py`;
- `src/model/qwen3_6_27b.cpp::bind_conv1d_view`;
- q5090 parser, fixture, and model-bind tests that assert conv1d shape or hidden allocation behavior.

线性/GDN 单层大小明细：

| tensor | qtype | estimated bytes |
|---|---|---:|
| `input_layernorm.weight` | `BF16_CTRL` | 0.009766 MiB |
| `linear_attn.A_log` | `FP32_CTRL` | 0.000183 MiB |
| `linear_attn.dt_bias` | `FP32_CTRL` | 0.000183 MiB |
| `linear_attn.conv1d.weight` | `BF16_CTRL` | 0.078125 MiB |
| `linear_attn.in_proj_a.weight` | `BF16_CTRL` | 0.468750 MiB |
| `linear_attn.in_proj_b.weight` | `BF16_CTRL` | 0.468750 MiB |
| `linear_attn.in_proj_qkv.weight::q rows[0:2048]` | `Q4G64_F16S` | 5.312500 MiB |
| `linear_attn.in_proj_qkv.weight::k rows[2048:4096]` | `Q4G64_F16S` | 5.312500 MiB |
| `linear_attn.in_proj_qkv.weight::v rows[4096:10240]` | `Q5G64_F16S` | 19.687500 MiB |
| `linear_attn.in_proj_z.weight` | `Q5G64_F16S` | 19.687500 MiB |
| `linear_attn.norm.weight` | `BF16_CTRL` | 0.000244 MiB |
| `linear_attn.out_proj.weight` | `Q5G64_F16S` | 19.687500 MiB |
| `post_attention_layernorm.weight` | `BF16_CTRL` | 0.009766 MiB |
| `mlp.gate_proj.weight` | `Q4G64_F16S` | 45.156250 MiB |
| `mlp.up_proj.weight` | `Q4G64_F16S` | 45.156250 MiB |
| `mlp.down_proj.weight` | `Q5G64_F16S` | 55.781250 MiB |
| **total** | - | **216.817017 MiB** |

### 5.1 `in_proj_qkv.weight` 离线切片规则

源 safetensors 为 fused tensor：

```text
linear_attn.in_proj_qkv.weight shape = [10240, 5120]
rows 0..2047     -> q  -> Q4G64_F16S
rows 2048..4095  -> k  -> Q4G64_F16S
rows 4096..10239 -> v  -> Q5G64_F16S
```

converter 输出三个 logical tensor entries：

```text
model.language_model.layers.{L}.linear_attn.in_proj_qkv.q
model.language_model.layers.{L}.linear_attn.in_proj_qkv.k
model.language_model.layers.{L}.linear_attn.in_proj_qkv.v
```

不要把不同 qtype 拼回同一个 payload；payload 粒度应该与 kernel 调用粒度一致。

---

## 6. Full attention 层详细模板

适用 layer：`3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63`。实际 tensor 前缀：

```text
model.language_model.layers.{L}.
```

| Tensor / logical slice | 逻辑/存储 shape | qtype | layout | 说明 |
|---|---:|---|---|---|
| `input_layernorm.weight` | `[5120]` | `BF16_CTRL` | contiguous | RMSNorm；不量化 |
| `self_attn.q_proj.weight::q` | per-head `[:, :256]` of `view[24,512,5120]` → `[6144,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | full attention query projection |
| `self_attn.q_proj.weight::gate` | per-head `[:, 256:]` of `view[24,512,5120]` → `[6144,5120]` | `Q5G64_F16S` | `TILE_N64_K64` | Qwen gated attention output gate，敏感 |

> ⚠️ `q_proj` is packed **per-head interleaved** as `[head_i: query(256) | gate(256)] ×24`
> (reference: `q_proj(h).view(-1, head_dim*2).chunk(2, dim=-1)`). The `::q` / `::gate`
> split is therefore `view([24,512,5120])` then take the first / last 256 rows of **each**
> head — **NOT** two contiguous `[0:6144]` / `[6144:12288]` blocks. The naive contiguous
> split scrambles heads (cos ≈ 0.04 vs the true query) and yields whitespace-only output.
> See `tools/q5090_convert/tensor_plan.py::attn_qproj_split` and
> `docs/qwen3.6-27b-architecture.md` §4.3 / §6.4. (GDN `in_proj_qkv` below **is** contiguous.)
| `self_attn.k_proj.weight` | `[1024,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | K projection |
| `self_attn.v_proj.weight` | `[1024,5120]` | `Q5G64_F16S` | `TILE_N64_K64` | V projection，敏感 |
| `self_attn.q_norm.weight` | `[256]` | `BF16_CTRL` | contiguous | head norm；不量化 |
| `self_attn.k_norm.weight` | `[256]` | `BF16_CTRL` | contiguous | head norm；不量化 |
| `self_attn.o_proj.weight` | `[5120,6144]` | `Q5G64_F16S` | `TILE_N64_K64` | attention output projection，敏感 |
| `post_attention_layernorm.weight` | `[5120]` | `BF16_CTRL` | contiguous | RMSNorm；不量化 |
| `mlp.gate_proj.weight` | `[17408,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | FFN gate |
| `mlp.up_proj.weight` | `[17408,5120]` | `Q4G64_F16S` | `TILE_N64_K64` | FFN up |
| `mlp.down_proj.weight` | `[5120,17408]` | `Q5G64_F16S` | `TILE_N64_K64` | FFN down，敏感 |

Full attention 单层大小明细：

| tensor | qtype | estimated bytes |
|---|---|---:|
| `input_layernorm.weight` | `BF16_CTRL` | 0.009766 MiB |
| `self_attn.q_proj.weight::q (per-head [:, :256])` | `Q4G64_F16S` | 15.937500 MiB |
| `self_attn.q_proj.weight::gate (per-head [:, 256:])` | `Q5G64_F16S` | 19.687500 MiB |
| `self_attn.k_proj.weight` | `Q4G64_F16S` | 2.656250 MiB |
| `self_attn.v_proj.weight` | `Q5G64_F16S` | 3.281250 MiB |
| `self_attn.q_norm.weight` | `BF16_CTRL` | 0.000488 MiB |
| `self_attn.k_norm.weight` | `BF16_CTRL` | 0.000488 MiB |
| `self_attn.o_proj.weight` | `Q5G64_F16S` | 19.687500 MiB |
| `post_attention_layernorm.weight` | `BF16_CTRL` | 0.009766 MiB |
| `mlp.gate_proj.weight` | `Q4G64_F16S` | 45.156250 MiB |
| `mlp.up_proj.weight` | `Q4G64_F16S` | 45.156250 MiB |
| `mlp.down_proj.weight` | `Q5G64_F16S` | 55.781250 MiB |
| **total** | - | **207.364258 MiB** |

### 6.1 `q_proj.weight` 离线切片规则

源 safetensors 中 `self_attn.q_proj.weight` 是 fused output：

```text
self_attn.q_proj.weight shape = [12288, 5120]
view[24, 512, 5120][:, :256, :]   -> query       -> [6144,5120] Q4G64_F16S
view[24, 512, 5120][:, 256:, :]   -> output gate -> [6144,5120] Q5G64_F16S
```

这里的 `512` 是每个 full-attention head 的 fused `[query(256) | gate(256)]` 输出宽度。
因此离线转换必须按 head 取前/后 256 行后再展平成 `[6144,5120]`；不能把
`[12288,5120]` 直接切成连续的 `[0:6144]` 和 `[6144:12288]` 两半。

converter 输出 logical tensor entries：

```text
model.language_model.layers.{L}.self_attn.q_proj.q
model.language_model.layers.{L}.self_attn.q_proj.gate
```

---

## 7. 文件格式建议

### 7.1 文件整体结构

```c
struct FileHeaderV1 {
    char     magic[16];        // "Q5090MIXEDV1\0"
    uint32_t version;          // 1
    uint32_t endian;           // 0x01020304
    uint32_t header_size;      // 4096 aligned
    uint32_t tensor_count;
    uint32_t module_count;      // 1..3: TEXT_CORE, optional MTP_DRAFT, optional VISION_ENCODER
    uint32_t layer_count;      // 64
    uint32_t flags;            // text_only/include_mtp/include_vision/calibrated/etc.
    uint64_t module_index_offset;
    uint64_t module_index_bytes;
    uint64_t tensor_index_offset;
    uint64_t tensor_index_bytes;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint8_t  sha256_safetensors_index[32];
    uint8_t  reserved[128];
};
```

每个 tensor entry：

```c
struct TensorEntryV1 {
    uint64_t name_hash;        // xxh3_64(canonical_name)
    uint32_t name_offset;      // string table offset
    uint16_t qtype;            // Q4/Q5/Q6/W8/BF16/FP32
    uint16_t layout;           // TILE_N64_K64/TILE_N64_K128/ROW_GROUPED_G64/CONTIGUOUS
    uint32_t ndim;
    uint32_t shape[4];         // logical shape
    uint32_t padded_shape[4];  // runtime shape
    uint32_t group_size;       // 64, 128, or 0
    uint64_t payload_offset;   // from file start, aligned
    uint64_t payload_bytes;
    uint32_t source_layer;     // 0..63 or UINT32_MAX
    uint32_t source_kind;      // enum projection type
    uint32_t source_row_start; // for fused source slicing
    uint32_t source_row_count;
    uint32_t crc32;
};
```

runtime 热路径不解析字符串。启动阶段根据 tensor index 构建固定 `LayerWeights[64]` 指针表。

### 7.2 Runtime 指针表

```c
struct QuantMat {
    void*    payload;          // interleaved scale+qdata tiles
    uint32_t N, K;
    uint32_t padded_N, padded_K;
    uint16_t qtype;
    uint16_t layout;
};

struct LinearGDNLayerWeights {
    bf16* input_ln;
    fp32* A_log;
    fp32* dt_bias;
    bf16* conv1d;
    bf16* in_proj_a;
    bf16* in_proj_b;
    QuantMat q_proj;
    QuantMat k_proj;
    QuantMat v_proj;
    QuantMat z_proj;
    bf16* linear_norm;
    QuantMat out_proj;
    bf16* post_ln;
    QuantMat mlp_gate;
    QuantMat mlp_up;
    QuantMat mlp_down;
};

struct FullAttnLayerWeights {
    bf16* input_ln;
    QuantMat q_proj_q;
    QuantMat q_proj_gate;
    QuantMat k_proj;
    QuantMat v_proj;
    bf16* q_norm;
    bf16* k_norm;
    QuantMat o_proj;
    bf16* post_ln;
    QuantMat mlp_gate;
    QuantMat mlp_up;
    QuantMat mlp_down;
};
```

---

## 8. Converter 规则

### 8.1 输入校验

1. `num_hidden_layers == 64`。
2. `layer_types[L]` 与本文一致，或 `full_attention_interval == 4` 且实际 weight_map 中 full/linear tensor 名称一致。
3. 每个 large tensor shape 与本文一致。
4. `lm_head.weight`、`model.language_model.embed_tokens.weight`、`model.language_model.norm.weight` 均存在。
5. 若输入 safetensors 包含 `model.visual.*` / `mtp.*`，converter 必须能按 flag 生成 optional segment；未启用时可跳过并写入 manifest 的 `disabled_segments`。

### 8.2 Scale 生成

最低实现：

```python
scale = max(abs(group)) / qmax
```

正式实现建议：

```python
# act_rms[k] 来自 calibration；无 calibration 时 act_rms=1
loss(scale) = sum_k act_rms[k]^2 * (W[row,k] - dequant(quant(W[row,k], scale), scale))^2
```

建议 scale 候选：

```text
maxabs_scale * {0.50, 0.55, 0.60, ..., 1.00, 1.05, 1.10, 1.15, 1.20}
```

优先把 calibration 预算用于 `Q4` tensor；`Q5/Q6` 多数情况下 maxabs_scale 足够稳。

---

## 9. Kernel 消费方式

### Decode GEMV，batch=1

```text
y[N] = Wq[N,K] @ x[K]
```

推荐 kernel：

- 一个 CTA 处理一个或多个 N64 tile。
- 遍历 K64 tile。
- `scale[64]` 128B 连续加载。
- 每 row 的 64 个 quant code 连续，避免跨 group gather。
- x[K64] 可预取到 shared/register。
- Q4/Q5/Q6/W8 分别专用 template，避免 runtime switch。

### Short prefill small-M GEMM

M 小时复用 GEMV 风格 kernel：

```text
Y[M,N] = A[M,K] @ Wq[N,K]^T
```

`TILE_N64_K64` 让 W 的 N/K tile 对齐，不需要第二份权重。

### Long prefill Tensor Core GEMM

长 prefill 时可在 kernel 内把 `TILE_N64_K64` 解包到 register/shared，再转 FP16/BF16 fragment 或走 int4 MMA 后乘 scale。该 layout 不一定是 Marlin 最优 prefill layout，但避免保存 decode/prefill 双权重；本项目应优先 decode。

---

## 10. Optional module segment 最终策略：MTP 与 Vision/mmproj

本节把 `mtp.*` 与多模态视觉路径作为可落地的 optional packed module。核心原则：**不改变 text core 的 v1 ABI**，而是在同一 packed 文件中增加可缺省、可懒加载、可单独校验的 module segment。

### 10.1 模块边界

Qwen3.6-27B 不是纯 text-only 权重：配置中 `language_model_only=false`，有 `vision_config`，且 `text_config.mtp_num_hidden_layers=1`。权重 index 中也包含 `model.visual.*` 与 `mtp.*`。最终文件格式按 module 切分：

| module | 是否必需 | 量化策略 | 推荐加载策略 |
|---|---:|---|---|
| `TEXT_CORE` | 必需 | `q5090_w4g64_mixed_v1` | 启动常驻 GPU |
| `MTP_DRAFT` | 可选 | `mtp_w8g128_v1` | speculative decode 启用时常驻；长上下文可禁用 |
| `VISION_ENCODER` | 可选 | `vision_q4mix_merger_w8g128_v1` | 多模态请求时 lazy load；生成 image/video embeds 后可释放 |

不要把 MTP 和 vision 强耦合进 text core fixed tensor list。没有 MTP/vision segment 时，runtime 仍必须能执行完全正确的 text-only 推理。

建议 packed file 的 module table 增加：

```c
struct Q5090ModuleRecord {
    uint32_t module_kind;      // 0=TEXT_CORE, 1=MTP_DRAFT, 2=VISION_ENCODER
    uint32_t module_version;   // TEXT=1, MTP=1, VISION=1
    uint64_t tensor_index_begin;
    uint64_t tensor_index_count;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint32_t load_policy;      // RESIDENT, LAZY_GPU, CPU_PINNED_THEN_GPU
    uint32_t flags;            // calibrated, can_free_after_prefill, requires_bias_kernel, etc.
};
```

---

### 10.2 MTP：最终默认 `mtp_w8g128_v1`

结论：**MTP 需要量化，但不要低于 8-bit 作为默认。**

MTP 的目标不是极限压缩，而是给主模型 verifier 提供高质量 draft/proposal token。只要 speculative decoding 有严格 verifier，MTP 量化误差主要影响 acceptance rate 和加速比；如果 acceptance 掉得多，省下来的 100 多 MiB 显存没有意义。因此最终版把前一版的 `mtp_q5safe_v1` 改为：

```text
MTP_DRAFT = mtp_w8g128_v1
```

#### 10.2.1 MTP tensor 列表

公开 index 中的 MTP tensor：

```text
mtp.fc.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
mtp.norm.weight
```

根据 text config 与 full-attention module 维度，MTP decoder layer 的主要矩阵维度应与一个 full-attention text layer 相同。`mtp.fc.weight` 用于 MTP 输入融合，预期 shape 为 `[5120, 10240]`。converter 必须从 safetensors 读取真实 shape 并 assert；若 shape 不匹配，不要硬编码继续转换。

#### 10.2.2 MTP qtype policy

| Tensor | 预期 shape | qtype | layout | 原因 |
|---|---:|---|---|---|
| `mtp.fc.weight` | `[5120,10240]` | `W8G128_F16S` | `TILE_N64_K128` | 输入融合层，直接影响 draft hidden；W8 稳定且只约 50.6 MiB |
| `mtp.layers.0.self_attn.q_proj.weight` | `[12288,5120]` | `W8G128_F16S` | `TILE_N64_K128` | 包含 query 与 output gate；MTP 中不再拆 Q/G 不同 bit |
| `mtp.layers.0.self_attn.k_proj.weight` | `[1024,5120]` | `W8G128_F16S` | `TILE_N64_K128` | 体积极小，不值得 Q4/Q5 |
| `mtp.layers.0.self_attn.v_proj.weight` | `[1024,5120]` | `W8G128_F16S` | `TILE_N64_K128` | value 路径敏感 |
| `mtp.layers.0.self_attn.o_proj.weight` | `[5120,6144]` | `W8G128_F16S` | `TILE_N64_K128` | residual output 路径敏感 |
| `mtp.layers.0.mlp.gate_proj.weight` | `[17408,5120]` | `W8G128_F16S` | `TILE_N64_K128` | MTP 只有一层，W8 成本可接受 |
| `mtp.layers.0.mlp.up_proj.weight` | `[17408,5120]` | `W8G128_F16S` | `TILE_N64_K128` | 同上 |
| `mtp.layers.0.mlp.down_proj.weight` | `[5120,17408]` | `W8G128_F16S` | `TILE_N64_K128` | FFN 聚合输出路径，优先保证 draft fidelity |
| `mtp.*norm*.weight` | small | `BF16_CTRL` | contiguous | norm 小且敏感 |
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | `BF16_CTRL` | contiguous | 小权重，不量化 |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | `BF16_CTRL` | contiguous | 小权重，不量化 |

MTP W8 的量化公式：

```text
qmax  = 127
group = W[row, k0:k0+128]
scale = max(abs(group)) / 127
q_i   = clamp(round(w_i / scale), -127, 127)
w_i≈  scale * q_i
```

v1 不使用 asymmetric INT8；不存 zero point。8-bit 动态范围足够，保留 symmetric 形式可以让 decode GEMV 和 small-M GEMM loader 更简单。

#### 10.2.3 MTP 体积估算

| MTP policy | 估算大小 | 说明 |
|---|---:|---|
| BF16 MTP | ~810 MiB / 0.791 GiB | 最稳但浪费显存 |
| `mtp_w8g128_v1` | ~411 MiB / 0.402 GiB | 最终默认；比 BF16 省约 399 MiB |
| `mtp_q5safe_v1` | ~266 MiB / 0.260 GiB | 只比 W8 再省约 146 MiB，但可能降低 acceptance |
| 复用 text mixed Q4/Q5 policy | ~240 MiB / 0.234 GiB | 不作为默认；acceptance 风险最大 |

最终建议：**默认保存并加载 `mtp_w8g128_v1`；如果 MTP 实测 acceptance 与 BF16 完全接近，后续可以实验 Q5，但不要把 Q5 作为 v1 runtime ABI。**

#### 10.2.4 MTP KV/state 预算

如果 MTP speculative decode 的实现需要为 `mtp.layers.0.self_attn` 保留 KV cache，则它是 **1 层 full attention KV**：

```text
K channel = 4 * 256 = 1024
V channel = 4 * 256 = 1024
BF16/FP16 MTP KV = 1 * (1024 + 1024) * 2 bytes = 4096 bytes/token
FP8 MTP KV       = 1 * (1024 + 1024) * 1 byte  = 2048 bytes/token
```

| context | MTP KV BF16/FP16 | MTP KV FP8 |
|---:|---:|---:|
| 4K | ~16 MiB | ~8 MiB |
| 32K | ~128 MiB | ~64 MiB |
| 128K | ~512 MiB | ~256 MiB |
| 256K | ~1.0 GiB | ~512 MiB |

Runtime policy：

```text
context <= 32K:   MTP 可默认启用，MTP KV BF16/FP16 或 FP8 都可
context >= 128K:  默认关闭 MTP，或强制 MTP KV FP8
context = 256K:   只有在 KV FP8 且 workspace 充足时启用 MTP
```

MTP 权重与 MTP KV 必须放在独立 arena，便于按请求开关和释放。

---

### 10.3 Vision / mmproj：最终默认 `vision_q4mix_merger_w8g128_v1`

Qwen3.6 权重里没有名为 `mm_projector` 或 `mmproj` 的 tensor。实际 vision-to-text bridge 是：

```text
model.visual.merger.linear_fc1.weight / bias
model.visual.merger.linear_fc2.weight / bias
model.visual.merger.norm.weight / bias
```

内部命名建议统一使用：

```text
VISION_MERGER
```

不要在 packed format 中使用模糊的 `mmproj` 名字。`VISION_MERGER` 的作用是把 vision hidden 投到 text hidden space：vision config 中 `hidden_size=1152`、`spatial_merge_size=2`，所以 merger 输入 context dim 为 `1152 * 2 * 2 = 4608`，最终 `out_hidden_size=5120`。

#### 10.3.1 Vision 是否需要量化

| 策略 | 推荐做法 | 适用场景 |
|---|---|---|
| vision weights 常驻 GPU | vision encoder blocks Q4/Q5 mixed，merger W8，pos BF16 | 多模态请求频繁，避免每次 CPU→GPU 上传 |
| vision weights lazy-load，用完释放 | 可以先 BF16 或使用同一 quantized segment | 多模态请求少，优先降低常驻显存 |
| 最高多模态质量 | blocks 可 Q4/Q5，merger BF16 fallback | OCR、图表、坐标、细节问答掉分时 |

最终默认不是 Q6 merger，而是：

```text
VISION_ENCODER = vision_q4mix_merger_w8g128_v1
```

原因：merger/mmproj 是视觉特征进入 text hidden space 的桥。W8 比 Q6 多约 10 MiB，但比 Q6 更稳；merger BF16 也只有约 85.5 MiB，可作为严格 fallback。

#### 10.3.2 Vision encoder block policy

Qwen3.6 vision config：`depth=27`、`hidden_size=1152`、`intermediate_size=4304`、`num_heads=16`、`out_hidden_size=5120`、`patch_size=16`、`temporal_patch_size=2`、`spatial_merge_size=2`。

每个 `model.visual.blocks.{B}`，`B=0..26`：

| Tensor | 逻辑 shape | qtype | layout | 说明 |
|---|---:|---|---|---|
| `model.visual.blocks.{B}.attn.qkv.weight` | `[3456,1152]` | `Q4G64_F16S` | `TILE_N64_K64` | q/k/v 合并矩阵；维度均 64 对齐 |
| `model.visual.blocks.{B}.attn.qkv.bias` | `[3456]` | `BF16_CTRL` | contiguous | vision 线性层有 bias，需单独加 |
| `model.visual.blocks.{B}.attn.proj.weight` | `[1152,1152]` | `Q5G64_F16S` | `TILE_N64_K64` | residual output 路径，升 Q5 |
| `model.visual.blocks.{B}.attn.proj.bias` | `[1152]` | `BF16_CTRL` | contiguous | bias |
| `model.visual.blocks.{B}.mlp.linear_fc1.weight` | `[4304,1152]` | `Q4G64_F16S` | `TILE_N64_K64` | MLP expand；N pad 4304→4352 |
| `model.visual.blocks.{B}.mlp.linear_fc1.bias` | `[4304]` | `BF16_CTRL` | contiguous | bias 逻辑长度 4304，不 pad 输出回写 |
| `model.visual.blocks.{B}.mlp.linear_fc2.weight` | `[1152,4304]` | `Q5G64_F16S` | `TILE_N64_K64` | MLP output residual 路径，K pad 4304→4352 |
| `model.visual.blocks.{B}.mlp.linear_fc2.bias` | `[1152]` | `BF16_CTRL` | contiguous | bias |
| `model.visual.blocks.{B}.norm1.weight/bias` | `[1152]` | `BF16_CTRL` | contiguous | LayerNorm 参数 |
| `model.visual.blocks.{B}.norm2.weight/bias` | `[1152]` | `BF16_CTRL` | contiguous | LayerNorm 参数 |

Vision 的 `intermediate_size=4304` 不是 64 的整数倍，tensor record 必须保存：

```c
uint32_t logical_N, logical_K;
uint32_t padded_N, padded_K;  // ceil(logical/64)*64 for Q4/Q5/Q6, ceil(logical_K/128)*128 for W8
```

kernel 对 padded tile 做无分支读取，输出写回时按 `logical_N` 截断。

#### 10.3.3 Patch embedding

```text
model.visual.patch_embed.proj.weight
model.visual.patch_embed.proj.bias
```

`Conv3d(in=3, out=1152, kernel=[2,16,16], stride=[2,16,16])` 可以离线 flatten 成一个 linear weight：

```text
logical shape = [1152, 3*2*16*16] = [1152,1536]
qtype        = Q5G64_F16S
layout       = TILE_N64_K64
bias         = BF16_CTRL
```

patch embedding 是视觉输入第一层。若 OCR/细粒度识别明显下降，可把 `patch_embed.proj.weight` 升为 `Q6G64_F16S` 或 BF16；它本身只有约 1.77M 参数，不值得为了极小省显存冒高风险。

#### 10.3.4 Positional embedding

```text
model.visual.pos_embed.weight  # [2304,1152]
```

推荐 `BF16_CTRL` 或专用 `BF16_TABLE`。不要 Q4。它只有约 5.1 MiB BF16，且参与视觉位置插值/加和，量化收益很小。

#### 10.3.5 Vision merger / mmproj policy

```text
model.visual.merger.linear_fc1.weight  # [4608,4608]
model.visual.merger.linear_fc1.bias    # [4608]
model.visual.merger.linear_fc2.weight  # [5120,4608]
model.visual.merger.linear_fc2.bias    # [5120]
model.visual.merger.norm.weight        # [1152]
model.visual.merger.norm.bias          # [1152]
```

最终默认：

| Tensor | qtype | layout | 原因 |
|---|---|---|---|
| `model.visual.merger.linear_fc1.weight` | `W8G128_F16S` | `TILE_N64_K128` | vision-to-text bridge 敏感；W8 起步 |
| `model.visual.merger.linear_fc2.weight` | `W8G128_F16S` | `TILE_N64_K128` | 最终进入 text hidden 的投影，Q4/Q5 不建议 |
| `model.visual.merger.*bias` | `BF16_CTRL` | contiguous | 小且直接加到输出 |
| `model.visual.merger.norm.weight/bias` | `BF16_CTRL` | contiguous | 小且敏感 |

Fallback：

```text
vision_merger_policy = BF16_STRICT
```

若多模态评测发现 W8 merger 仍影响 OCR、图表、坐标、细节问答，直接回退 BF16。不要尝试 Q4 merger。

#### 10.3.6 Vision 体积估算

| Vision policy | 估算大小 | 说明 |
|---|---:|---|
| 全 BF16 vision | ~879 MiB / 0.858 GiB | 最简单、质量参照；不建议长上下文常驻 |
| `vision_q4mix_merger_bf16_v1` | ~321 MiB / 0.314 GiB | blocks mixed Q4/Q5，merger BF16 |
| `vision_q4mix_merger_q6_v1` | ~269 MiB / 0.263 GiB | 旧推荐；仍可作为实验变体 |
| `vision_q4mix_merger_w8g128_v1` | ~279 MiB / 0.272 GiB | 最终默认；比 Q6 只多约 10 MiB，但更稳 |

推荐 runtime 在多模态 prefill 阶段执行：

```text
1. lazy load VISION_ENCODER weights to a temporary GPU arena
2. run image/video encoder, produce BF16 image_embeds/video_embeds in text hidden space
3. scatter image/video embeds into text input embeddings
4. release VISION_ENCODER arena before long text prefill/decode, unless 后续请求会复用
```

如果产品要求所有权重常驻 GPU，则使用 quantized vision policy；如果多模态请求稀疏，vision segment 可以放 CPU pinned memory，需要时 cudaMemcpy 到临时 arena。

---

## 11. 最终显存预算

### 11.1 权重常驻预算

| 组合 | text core | MTP | vision | 总权重常驻估算 |
|---|---:|---:|---:|---:|
| text only `q5090_w4g64_mixed_v1` | ~15.254 GiB | 0 | 0 | ~15.254 GiB |
| text + `mtp_w8g128_v1` | ~15.254 GiB | ~0.402 GiB | 0 | ~15.655 GiB |
| text + MTP W8 + `vision_q4mix_merger_w8g128_v1` | ~15.254 GiB | ~0.402 GiB | ~0.272 GiB | ~15.928 GiB |
| text + MTP W8 + vision BF16 | ~15.254 GiB | ~0.402 GiB | ~0.858 GiB | ~16.514 GiB |
| text + MTP BF16 + vision BF16 | ~15.254 GiB | ~0.791 GiB | ~0.858 GiB | ~16.903 GiB |

最终默认 all-segment quantized 常驻权重仍约 **15.93 GiB**，接近 text-only 的 16 GiB 预留线，不会明显挤压 32GB 卡上的 KV cache 与 workspace。

### 11.2 Full attention KV cache 预算

Qwen3.6 text core 只有 16 层 full attention。每个 full attention 层：

```text
K channel = 4 * 256 = 1024
V channel = 4 * 256 = 1024
BF16/FP16 per layer = (1024 + 1024) * 2 = 4096 bytes/token
16 layers BF16/FP16 = 65536 bytes/token = 64 KiB/token
16 layers FP8       = 32768 bytes/token = 32 KiB/token
```

| context | text KV BF16/FP16 | text KV FP8 | MTP KV BF16/FP16 extra | MTP KV FP8 extra |
|---:|---:|---:|---:|---:|
| 4K | ~256 MiB | ~128 MiB | ~16 MiB | ~8 MiB |
| 32K | ~2.0 GiB | ~1.0 GiB | ~128 MiB | ~64 MiB |
| 128K | ~8.0 GiB | ~4.0 GiB | ~512 MiB | ~256 MiB |
| 256K | ~16.0 GiB | ~8.0 GiB | ~1.0 GiB | ~512 MiB |

### 11.3 GDN state / activation / workspace / runtime buffer

GDN recurrent state 不按普通 KV cache 估算。按 48 个 linear attention 层估算：

```text
每层 recurrent state 近似量级 = 48 value heads * 128 key dim * 128 value dim
                         = 786,432 elements/layer
48 层 = 37.75M elements
BF16 约 72 MiB；FP32 约 144 MiB
conv_state 约数 MiB 量级
```

实际 kernel 可能保留额外 scratch、alignment、streaming buffer，建议 runtime 预留：

| 项目 | 建议预算 |
|---|---:|
| GDN recurrent + conv state | 128–256 MiB |
| activation buffer，decode | 128–512 MiB |
| activation buffer，prefill | 512 MiB–2 GiB，取决于 chunk/M |
| W4/W5/W6/W8 dequant workspace | 256 MiB–1 GiB |
| CUDA graph / runtime buffer | 256–768 MiB |
| allocator fragmentation / alignment | 512 MiB–1.5 GiB |

### 11.4 RTX 5090 32GB 现实结论

| context | KV dtype | text+MTP+vision 默认权重 | KV | 其它 state/workspace 预留 | 结论 |
|---:|---|---:|---:|---:|---|
| 4K | BF16/FP16 | ~15.93 GiB | ~0.25 GiB | 2–4 GiB | 很宽松 |
| 32K | BF16/FP16 | ~15.93 GiB | ~2.0 GiB | 2–5 GiB | 宽松 |
| 128K | BF16/FP16 | ~15.93 GiB | ~8.0 GiB | 3–5 GiB | 可行但需控制 workspace；vision 最好 lazy/free |
| 128K | FP8 | ~15.93 GiB | ~4.0 GiB | 3–5 GiB | 更稳 |
| 256K | BF16/FP16 | ~15.93 GiB | ~16.0 GiB | 3–5 GiB | 不建议；容易超过 32GB |
| 256K | FP8 | ~15.93 GiB | ~8.0 GiB | 3–5 GiB | 可作为长上下文目标；建议关闭 MTP 或 MTP KV FP8 |

---

## 12. Converter 最终实现 checklist

### 12.1 Text core checklist

对每个 Linear/GDN layer `L` in `0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, 44-46, 48-50, 52-54, 56-58, 60-62`：

```text
[ ] read model.language_model.layers.{L}.input_layernorm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.linear_attn.A_log -> FP32_CTRL
[ ] read model.language_model.layers.{L}.linear_attn.dt_bias -> FP32_CTRL
[ ] read model.language_model.layers.{L}.linear_attn.conv1d.weight -> BF16_CTRL contiguous [10240,4,1]
[ ] read model.language_model.layers.{L}.linear_attn.in_proj_a.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.linear_attn.in_proj_b.weight -> BF16_CTRL
[ ] split model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight rows[0:2048] -> Q4G64_F16S .q
[ ] split model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight rows[2048:4096] -> Q4G64_F16S .k
[ ] split model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight rows[4096:10240] -> Q5G64_F16S .v
[ ] read model.language_model.layers.{L}.linear_attn.in_proj_z.weight -> Q5G64_F16S
[ ] read model.language_model.layers.{L}.linear_attn.norm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.linear_attn.out_proj.weight -> Q5G64_F16S
[ ] read model.language_model.layers.{L}.post_attention_layernorm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.mlp.gate_proj.weight -> Q4G64_F16S
[ ] read model.language_model.layers.{L}.mlp.up_proj.weight -> Q4G64_F16S
[ ] read model.language_model.layers.{L}.mlp.down_proj.weight -> Q5G64_F16S
```

对每个 full attention layer `L` in `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63`：

```text
[ ] read model.language_model.layers.{L}.input_layernorm.weight -> BF16_CTRL
[ ] split model.language_model.layers.{L}.self_attn.q_proj.weight per-head view[24,512,5120][:, :256] -> Q4G64_F16S .q
[ ] split model.language_model.layers.{L}.self_attn.q_proj.weight per-head view[24,512,5120][:, 256:] -> Q5G64_F16S .gate
[ ] read model.language_model.layers.{L}.self_attn.k_proj.weight -> Q4G64_F16S
[ ] read model.language_model.layers.{L}.self_attn.v_proj.weight -> Q5G64_F16S
[ ] read model.language_model.layers.{L}.self_attn.q_norm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.self_attn.k_norm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.self_attn.o_proj.weight -> Q5G64_F16S
[ ] read model.language_model.layers.{L}.post_attention_layernorm.weight -> BF16_CTRL
[ ] read model.language_model.layers.{L}.mlp.gate_proj.weight -> Q4G64_F16S
[ ] read model.language_model.layers.{L}.mlp.up_proj.weight -> Q4G64_F16S
[ ] read model.language_model.layers.{L}.mlp.down_proj.weight -> Q5G64_F16S
```

### 12.2 MTP checklist

```text
[ ] if --include-mtp:
[ ]   create MTP_DRAFT segment with policy mtp_w8g128_v1
[ ]   read mtp.fc.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.self_attn.q_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.self_attn.k_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.self_attn.v_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.self_attn.o_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.mlp.gate_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.mlp.up_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.layers.0.mlp.down_proj.weight -> W8G128_F16S TILE_N64_K128
[ ]   read mtp.pre_fc_norm_embedding.weight -> BF16_CTRL
[ ]   read mtp.pre_fc_norm_hidden.weight -> BF16_CTRL
[ ]   read mtp.layers.0.input_layernorm.weight -> BF16_CTRL
[ ]   read mtp.layers.0.self_attn.q_norm.weight -> BF16_CTRL
[ ]   read mtp.layers.0.self_attn.k_norm.weight -> BF16_CTRL
[ ]   read mtp.layers.0.post_attention_layernorm.weight -> BF16_CTRL
[ ]   read mtp.norm.weight -> BF16_CTRL
[ ] else:
[ ]   write manifest disabled_segments += ["MTP_DRAFT"]
```

### 12.3 Vision/mmproj checklist

```text
[ ] if --include-vision:
[ ]   create VISION_ENCODER segment with policy vision_q4mix_merger_w8g128_v1
[ ]   for B in 0..26:
[ ]     read model.visual.blocks.{B}.attn.qkv.weight -> Q4G64_F16S TILE_N64_K64
[ ]     read model.visual.blocks.{B}.attn.qkv.bias -> BF16_CTRL
[ ]     read model.visual.blocks.{B}.attn.proj.weight -> Q5G64_F16S TILE_N64_K64
[ ]     read model.visual.blocks.{B}.attn.proj.bias -> BF16_CTRL
[ ]     read model.visual.blocks.{B}.mlp.linear_fc1.weight -> Q4G64_F16S TILE_N64_K64, pad N 4304->4352
[ ]     read model.visual.blocks.{B}.mlp.linear_fc1.bias -> BF16_CTRL
[ ]     read model.visual.blocks.{B}.mlp.linear_fc2.weight -> Q5G64_F16S TILE_N64_K64, pad K 4304->4352
[ ]     read model.visual.blocks.{B}.mlp.linear_fc2.bias -> BF16_CTRL
[ ]     read model.visual.blocks.{B}.norm1.weight/bias -> BF16_CTRL
[ ]     read model.visual.blocks.{B}.norm2.weight/bias -> BF16_CTRL
[ ]   flatten model.visual.patch_embed.proj.weight -> [1152,1536] -> Q5G64_F16S
[ ]   read model.visual.patch_embed.proj.bias -> BF16_CTRL
[ ]   read model.visual.pos_embed.weight -> BF16_TABLE
[ ]   read model.visual.merger.linear_fc1.weight -> W8G128_F16S TILE_N64_K128
[ ]   read model.visual.merger.linear_fc1.bias -> BF16_CTRL
[ ]   read model.visual.merger.linear_fc2.weight -> W8G128_F16S TILE_N64_K128
[ ]   read model.visual.merger.linear_fc2.bias -> BF16_CTRL
[ ]   read model.visual.merger.norm.weight/bias -> BF16_CTRL
[ ] else:
[ ]   write manifest disabled_segments += ["VISION_ENCODER"]
```

---

## 13. Kernel 消费最终建议

### 13.1 Decode GEMV

Text core：

```text
Q4/Q5/Q6 TILE_N64_K64 loader
- one warp or warp-group handles N tile / row subset
- load scale[64] once per K64 tile
- unpack qdata to int, convert to half/bfloat16, multiply scale
- accumulate into FP32 or FP16 accumulator depending on kernel target
```

MTP / vision merger：

```text
W8 TILE_N64_K128 loader
- no bit unpack, only int8 -> half/bfloat16 convert
- scale[64] per K128 tile
- same pointer-table infrastructure as text core
```

### 13.2 Prefill GEMM

单用户场景优先 decode，但 prefill 不能太差。最终策略：

```text
short prefill, small M:
  use same packed weight layout, dequant in registers/shared memory

long prefill:
  Q4/Q5/Q6/W8 dequant tile -> MMA fragment/shared staging
  不保存第二套权重 layout
```

不建议为了 prefill 再保存一份 Tensor Core 专用权重副本。32GB 单卡上，第二份 layout 会直接挤压 KV cache，违背长上下文目标。若 long prefill 性能不足，优先优化 tile dequant path 和 chunk size，而不是改格式。

### 13.3 Layout 取舍

| 问题 | 最终决策 |
|---|---|
| group size | text Q4/Q5/Q6 用 G64；MTP/merger W8 用 G128 |
| symmetric/asymmetric | 全部 symmetric；不存 zero/min |
| scale dtype | FP16；不做二级 scale 量化 |
| decode vs prefill | 优先 decode；prefill 复用同一 layout |
| N/K padding | 大矩阵 pad 到 tile；record 保留 logical shape |
| runtime metadata | 启动时解析一次，hot path 只用 typed pointer table |
| 文件与 runtime layout | 同一套；离线 converter 完成 packing |

---

## 14. Calibration 与评估

### 14.1 Text core calibration

文件格式不依赖 calibration，但建议 converter 支持 calibration scale search：

```text
v1 baseline:
  maxabs scale per group

v1 calibrated:
  activation-RMS weighted MSE scale search
  或 AWQ-style clipping / imatrix-like statistics
  不改变 payload layout，只改变 scale 生成
```

优先把 calibration 预算用于：

```text
1. Q4 tensors: mlp.gate_proj, mlp.up_proj, attention/GDN q/k
2. first 2 layers and last 2 layers
3. full attention layers
4. GDN q/k/v/z/out path
```

### 14.2 MTP evaluation

MTP 的指标不要只看 perplexity；重点看 speculative decoding：

```text
BF16 MTP vs W8 MTP draft token top-1 agreement
draft logits KL vs BF16 MTP
acceptance_rate@draft_len=2/4/8
average accepted tokens / verification step
end-to-end tokens/s
```

合格线建议：

```text
acceptance rate 下降 <= 1-2 个百分点
average accepted tokens 下降 <= 3%
如果不合格，优先把 mtp.fc.weight 和 mtp.layers.0.mlp.down_proj.weight 回退 BF16，
不要直接修改 MTP segment ABI。
```

### 14.3 Vision/mmproj calibration

Vision calibration 必须使用 image/video 数据，而不是纯文本 imatrix：

```text
1. OCR：小字、表格、票据、截图、代码截图
2. 图表：坐标轴、折线/柱状图、legend、多子图
3. 空间定位：left/right/top/bottom、计数、颜色、位置关系
4. 多图对比：两张或多张图的差异与引用
5. 视频/多帧：短视频关键帧、时间顺序、动作变化
```

合格线建议：

```text
vision_q4mix_merger_w8g128_v1 相对 BF16 vision：
- OCR / chart / spatial QA 主观失败率不能明显增加
- multimodal eval 分数下降尽量 <= 1-2%
- text-only eval 必须 bit-identical 或在浮点非确定性范围内一致，因为 vision module 不参与 text-only path
```

---

## 15. Manifest 示例

```json
{
  "format": "q5090_w4g64_mixed_v1_final",
  "model": "Qwen/Qwen3.6-27B",
  "format_version": 1,
  "segments": [
    {
      "kind": "TEXT_CORE",
      "policy": "q5090_w4g64_mixed_v1",
      "resident": true,
      "estimated_weight_size_gib": 15.254,
      "effective_text_bpw": 4.872
    },
    {
      "kind": "MTP_DRAFT",
      "policy": "mtp_w8g128_v1",
      "resident_default": true,
      "estimated_weight_size_gib": 0.402,
      "kv_policy_default": "fp8_when_context_ge_128k"
    },
    {
      "kind": "VISION_ENCODER",
      "policy": "vision_q4mix_merger_w8g128_v1",
      "resident_default": false,
      "estimated_weight_size_gib": 0.272,
      "fallback": "vision_merger_bf16_strict"
    }
  ],
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "num_hidden_layers": 64,
  "full_attention_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63],
  "linear_attention_layers": "0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, 44-46, 48-50, 52-54, 56-58, 60-62",
  "qtypes": ["Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G128_F16S", "BF16_CTRL", "FP32_CTRL"],
  "zero_point": false,
  "layouts": ["TILE_N64_K64", "TILE_N64_K128", "ROW_GROUPED_G64", "CONTIGUOUS"],
  "alignment": {
    "file_header": 4096,
    "tensor_payload": 256,
    "recommended_tensor_payload": 4096
  }
}
```

---

## 16. 最终锁定建议

启动实现时锁定以下 ABI：

```text
TEXT_CORE:
  q5090_w4g64_mixed_v1
  Q4/Q5/Q6 G64 + FP16 scale + symmetric + no zero/min

MTP_DRAFT:
  mtp_w8g128_v1
  W8 G128 + FP16 scale + symmetric + no zero/min

VISION_ENCODER:
  vision_q4mix_merger_w8g128_v1
  blocks Q4/Q5 mixed
  patch Q5
  pos BF16
  merger/mmproj W8, BF16 fallback
```

不建议的路线：

```text
naive all-Q4:
  down_proj、v/o、lm_head、embedding、GDN value/output 路径质量风险过高。

直接用 GGUF Q4_K_M runtime layout:
  metadata 解码复杂、scale/min packing 不适合热路径、CUDA tile/coalesced loading 不理想，仍可能需要 repack。

MTP Q4/Q5 默认:
  MTP 是 draft fidelity 模块，省 0.1-0.2 GiB 不值得冒 acceptance rate 风险。

vision merger/mmproj Q4:
  视觉特征进入 text hidden 的桥，质量风险明显大于显存收益。
```

最终工程路线：

```text
1. 先实现 TEXT_CORE q5090_w4g64_mixed_v1，跑通 text-only decode/prefill。
2. 同时在文件 ABI 中保留 module table，不要后期再改 header。
3. 第二阶段加入 MTP_DRAFT mtp_w8g128_v1，验收 acceptance/token speed。
4. 第三阶段加入 VISION_ENCODER vision_q4mix_merger_w8g128_v1，先支持 lazy load/free。
5. calibration 只改 scale，不改 payload layout。
6. sensitivity fallback 通过 per-tensor qtype override 或 BF16 fallback 完成，不改主 ABI。
```

这个最终格式的复杂度是值得的：text core 的 4.87 bpw 接近 Q4_K_M 的质量目标，runtime layout 又比 GGUF 更直接；MTP 与 vision/mmproj 用独立 segment 解决后期扩展问题，不会破坏已经优化的 decode hot path。

---

## 17. 参考来源与校验点

实现时应至少校验以下公开文件：

```text
Qwen/Qwen3.6-27B config.json
- architecture: Qwen3_5ForConditionalGeneration
- language_model_only=false
- text_config: hidden_size=5120, intermediate_size=17408, num_hidden_layers=64
- full_attention_interval=4, layer_types length=64
- mtp_num_hidden_layers=1
- vision_config: depth=27, hidden_size=1152, intermediate_size=4304, out_hidden_size=5120

Qwen/Qwen3.6-27B model.safetensors.index.json
- model.language_model.* tensor names
- lm_head.weight
- mtp.* tensor names
- model.visual.blocks.* tensor names
- model.visual.merger.* tensor names

Transformers qwen3_5 implementation
- Qwen3_5GatedDeltaNet uses in_proj_qkv/in_proj_z/in_proj_a/in_proj_b and splits q/k/value
- Qwen3_5ForCausalLM ignores mtp.* and model.visual.* for text-only path
```
