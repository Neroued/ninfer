# QUS Packed Weight File Format `q5090_w4g64_mixed_v1` (binary spec)

This document is the **M2.8/M3 canonical binary contract** that the offline converter
(`tools/q5090_convert/`) must produce for Qwen3.6-27B. It finalizes section 7 of
[qwen3_6_27b_q5090_final_quant_format_v1.md](qwen3_6_27b_q5090_final_quant_format_v1.md)
(the quantization *policy*) into precise byte offsets so the C++ runtime can mmap and
consume it with no runtime repacking.

The policy document decides *which qtype/layout each tensor gets*. This document decides
*how those bytes are laid out on disk*. Read them together.

> This is the canonical packed-weight ABI for the C++ runtime. The runtime consumes this format
> directly; there is no alternate in-tree weight-file path.
>
> Implementation status, 2026-06-27: P1 implements the canonical conv1d sync across converter output,
> q5090 fixtures, runtime binding, and tests. Official M2.8/M3 q5090 artifacts must use `[10240,4,1]`.
> Existing raw `[10240,1,4]` q5090 files are legacy pre-M2.8 artifacts. They must be regenerated before
> they are used as official M2.8/M3 baseline inputs.

## 0. Conventions

- All integers are **little-endian**.
- All reserved bytes are **zero**.
- Offsets are **absolute** file offsets unless stated otherwise.
- File starts with a 4096-byte header. Module index, tensor index and string table follow.
  The payload region starts at a 4096-byte boundary; each tensor payload is **256-byte aligned**.
- **No baked runtime math transforms.** Weights are stored without mathematical folding:
  `A_log`/`dt_bias` upcast to FP32 (they are bf16 on disk), RMSNorm weights stored raw (no `+1`), and
  folds such as `A = -exp(A_log)` remain the runtime's job.
- **Canonical TEXT_CORE conv1d layout.** For M2.8/M3 canonical artifacts,
  `linear_attn.conv1d.weight` is a contiguous BF16 tensor with logical shape `[10240,4,1]`
  (`[conv_dim,gdn_conv_k,1]`). This is a TEXT_CORE tensor logical-shape policy, not a binary container
  ABI redesign. Existing raw `[10240,1,4]` q5090 files are legacy pre-M2.8 artifacts. They must be
  regenerated before they are used as official M2.8/M3 baseline inputs.

## 1. File structure

```text
+--------------------------------------------------+ 0
| FileHeader (4096 bytes)                          |
+--------------------------------------------------+ header_size (4096)
| ModuleRecord[module_count]   (64 bytes each)     |
+--------------------------------------------------+ tensor_index_offset
| TensorEntry[tensor_count]    (128 bytes each)    |
+--------------------------------------------------+ string_table_offset
| String table (NUL-terminated tensor names)       |
+--------------------------------------------------+ payload_offset (4096-aligned)
| Tensor payloads (each 256-aligned, no overlap)   |
+--------------------------------------------------+ file_size
```

Modules are stored as **contiguous ranges** of the tensor index: all TEXT_CORE entries first,
then MTP_DRAFT, then VISION_ENCODER. Each `ModuleRecord` points at its `[begin, begin+count)`
slice of the tensor index and at the byte span of its payloads.

## 2. FileHeader (4096 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 16 | char | `magic` = `"Q5090MIXEDV1\0\0\0\0"` |
| 16 | 4 | u32 | `version` = 1 |
| 20 | 4 | u32 | `endian` = `0x01020304` |
| 24 | 4 | u32 | `header_size` = 4096 |
| 28 | 4 | u32 | `tensor_count` |
| 32 | 4 | u32 | `module_count` (1..3) |
| 36 | 4 | u32 | `layer_count` = 64 |
| 40 | 4 | u32 | `flags` (bit0 TEXT, bit1 MTP, bit2 VISION, bit3 calibrated) |
| 44 | 4 | u32 | reserved0 = 0 |
| 48 | 8 | u64 | `module_index_offset` |
| 56 | 8 | u64 | `module_index_bytes` |
| 64 | 8 | u64 | `tensor_index_offset` |
| 72 | 8 | u64 | `tensor_index_bytes` |
| 80 | 8 | u64 | `string_table_offset` |
| 88 | 8 | u64 | `string_table_bytes` |
| 96 | 8 | u64 | `payload_offset` |
| 104 | 8 | u64 | `payload_bytes` |
| 112 | 4 | u32 | `hidden_size` (5120) |
| 116 | 4 | u32 | `intermediate_size` (17408) |
| 120 | 4 | u32 | `vocab_size` (248320) |
| 124 | 4 | u32 | `num_attention_heads` (24) |
| 128 | 4 | u32 | `num_key_value_heads` (4) |
| 132 | 4 | u32 | `head_dim` (256) |
| 136 | 4 | u32 | `gdn_key_heads` (16) |
| 140 | 4 | u32 | `gdn_value_heads` (48) |
| 144 | 4 | u32 | `gdn_key_head_dim` (128) |
| 148 | 4 | u32 | `gdn_value_head_dim` (128) |
| 152 | 4 | u32 | `gdn_conv_width` (4) |
| 156 | 4 | u32 | `full_attention_interval` (4) |
| 160 | 4 | u32 | `max_position_embeddings` (262144) |
| 164 | 4 | u32 | reserved1 = 0 |
| 168 | 32 | u8 | `sha256_safetensors_index` (sha256 of `model.safetensors.index.json`) |
| 200 | 3896 | u8 | reserved zero |

## 3. ModuleRecord (64 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `module_kind` (0=TEXT_CORE, 1=MTP_DRAFT, 2=VISION_ENCODER) |
| 4 | 4 | u32 | `module_version` = 1 |
| 8 | 8 | u64 | `tensor_index_begin` |
| 16 | 8 | u64 | `tensor_index_count` |
| 24 | 8 | u64 | `payload_offset` (first tensor payload of module) |
| 32 | 8 | u64 | `payload_bytes` (span of module payloads incl. inter-tensor padding) |
| 40 | 4 | u32 | `load_policy` (0=RESIDENT, 1=LAZY_GPU, 2=CPU_PINNED_THEN_GPU) |
| 44 | 4 | u32 | `flags` |
| 48 | 16 | u8 | reserved zero |

Default `load_policy`: TEXT=RESIDENT, MTP=RESIDENT, VISION=LAZY_GPU.

## 4. TensorEntry (128 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `name_offset` (into string table) |
| 4 | 4 | u32 | `name_len` (bytes, excluding NUL) |
| 8 | 8 | u64 | `name_hash` (FNV-1a-64 of the canonical name) |
| 16 | 2 | u16 | `qtype` |
| 18 | 2 | u16 | `layout` |
| 20 | 2 | u16 | `module_kind` |
| 22 | 2 | u16 | `ndim` |
| 24 | 16 | u32[4] | `shape` (logical; trailing dims = 1) |
| 40 | 16 | u32[4] | `padded_shape` (runtime/tiled shape) |
| 56 | 4 | u32 | `group_size` (64, 128, or 0) |
| 60 | 2 | u16 | `scale_dtype` (0=none, 1=FP16) |
| 62 | 2 | u16 | reserved0 |
| 64 | 8 | u64 | `payload_offset` (absolute, 256-aligned) |
| 72 | 8 | u64 | `payload_bytes` |
| 80 | 4 | u32 | `source_layer` (0..63, or 0xFFFFFFFF for globals) |
| 84 | 4 | u32 | `source_kind` (see section 5) |
| 88 | 4 | u32 | `crc32` (zlib CRC-32 over the payload bytes) |
| 92 | 4 | u32 | reserved1 |
| 96 | 32 | u8 | reserved zero |

The runtime hot path does not parse strings; it builds a fixed `LayerWeights[64]` pointer
table once at load using `source_kind` + `source_layer` (or the canonical name).

## 5. Enums

`qtype` (u16):

| value | name | bits | group | signed range |
|---:|---|---:|---:|---|
| 0 | `Q4G64_F16S` | 4 | 64 | [-8, 7] |
| 1 | `Q5G64_F16S` | 5 | 64 | [-16, 15] |
| 2 | `Q6G64_F16S` | 6 | 64 | [-32, 31] |
| 3 | `W8G128_F16S` | 8 | 128 | [-127, 127] |
| 4 | `BF16_CTRL` | 16 | - | bfloat16 |
| 5 | `FP32_CTRL` | 32 | - | float32 |

`layout` (u16): `0=TILE_N64_K64`, `1=TILE_N64_K128`, `2=ROW_GROUPED_G64`, `3=CONTIGUOUS`.

`module_kind` (u16): `0=TEXT_CORE`, `1=MTP_DRAFT`, `2=VISION_ENCODER`.

`scale_dtype` (u16): `0=none`, `1=FP16`.

`source_kind` (u32): informational projection identity used to build pointer tables.

```text
0  OTHER                     30 ATTN_Q                60 VIS_PATCH_EMBED
1  EMBED                     31 ATTN_GATE             61 VIS_PATCH_EMBED_BIAS
2  LM_HEAD                   32 ATTN_K                62 VIS_POS_EMBED
3  FINAL_NORM                33 ATTN_V                63 VIS_BLOCK_QKV
4  INPUT_LAYERNORM           34 ATTN_Q_NORM          64 VIS_BLOCK_QKV_BIAS
5  POST_ATTN_LAYERNORM       35 ATTN_K_NORM          65 VIS_BLOCK_PROJ
10 GDN_A_LOG                 36 ATTN_O               66 VIS_BLOCK_PROJ_BIAS
11 GDN_DT_BIAS               40 MLP_GATE             67 VIS_BLOCK_FC1
12 GDN_CONV1D                41 MLP_UP               68 VIS_BLOCK_FC1_BIAS
13 GDN_IN_PROJ_A             42 MLP_DOWN             69 VIS_BLOCK_FC2
14 GDN_IN_PROJ_B             50 MTP_FC               70 VIS_BLOCK_FC2_BIAS
15 GDN_IN_PROJ_Q             51 MTP_PRE_FC_NORM_EMB  71 VIS_BLOCK_NORM1_W
16 GDN_IN_PROJ_K             52 MTP_PRE_FC_NORM_HID  72 VIS_BLOCK_NORM1_B
17 GDN_IN_PROJ_V             53 MTP_NORM             73 VIS_BLOCK_NORM2_W
18 GDN_IN_PROJ_Z                                     74 VIS_BLOCK_NORM2_B
19 GDN_NORM                                          75 VIS_MERGER_FC1
20 GDN_OUT_PROJ                                      76 VIS_MERGER_FC1_BIAS
                                                     77 VIS_MERGER_FC2
                                                     78 VIS_MERGER_FC2_BIAS
                                                     79 VIS_MERGER_NORM_W
                                                     80 VIS_MERGER_NORM_B
```

## 6. Quantization math (weight-only, symmetric, no zero point)

For each group of `group_size` consecutive K-elements of one row:

```text
qmax  = 2^(bits-1) - 1            # Q4=7, Q5=15, Q6=31, W8=127
scale = max(abs(group)) / qmax    # computed in fp32
scale16 = fp16(scale)             # the value stored AND used to quantize
q_i   = clamp(round(w_i / scale16), qmin, qmax)
```

- `qmin = -(qmax+1)` for Q4/Q5/Q6; `qmin = -127` for W8.
- Rounding is round-half-to-even. The converter quantizes with the **fp16-rounded** scale so
  on-disk codes are optimal for the exact scale the runtime reads back.
- All-zero group: `scale = 0`, all codes `0` (dequant yields 0). A nonzero group whose scale
  would underflow fp16 to zero is bumped to the smallest positive fp16 subnormal.
- Dequant: `w_i ~= scale16 * q_i`.

## 7. Payload encodings

`bits_per_row = ceil(group_size * bits / 8)`: Q4=32, Q5=40, Q6=48 bytes per 64-element group.

### 7.1 Code bit-packing (Q4/Q5/Q6)

Within one 64-code group, each signed code is stored as a `bits`-bit two's-complement value.
The 64 values are concatenated **LSB-first**: bit `j` (`0`=LSB) of code `c` lands at stream
position `c*bits + j`, then the stream is packed into bytes LSB-first (byte 0 holds stream bits
0..7 with bit 0 in the byte's least-significant position). This equals
`numpy.packbits(..., bitorder="little")`. The decoder sign-extends from bit `bits-1`.

For Q4 this reduces to standard nibble packing: byte `j` = `code[2j] & 0xF | ((code[2j+1] & 0xF) << 4)`.

### 7.2 `TILE_N64_K64` (Q4/Q5/Q6)

Logical `[N,K]` (PyTorch `nn.Linear.weight`). `N` padded to a multiple of 64, `K` to 64.

```text
for n_tile in 0 .. N/64 - 1:
  for k_tile in 0 .. K/64 - 1:
    fp16 scale[64]                 # 128 B, one scale per row for this K64 group
    for row in 0 .. 63:
      u8 packed[bits_per_row]      # row's 64 codes
```

Tile bytes: Q4=2176, Q5=2688, Q6=3200.

### 7.3 `TILE_N64_K128` (W8)

`N` padded to 64, `K` to 128.

```text
for n_tile in 0 .. N/64 - 1:
  for k_tile in 0 .. K/128 - 1:
    fp16 scale[64]                 # 128 B
    int8 qdata[64][128]            # 8192 B, row-major in tile
```

Tile bytes: 8320.

### 7.4 `ROW_GROUPED_G64` (embedding, Q6)

`model.language_model.embed_tokens.weight`, logical `[vocab, hidden]`, `hidden` padded to 64.
Per-row groups (no 64-row tiling) so a single token lookup touches only its own row:

```text
for token_id in 0 .. vocab-1:
  for k_tile in 0 .. hidden/64 - 1:
    fp16 scale                     # 2 B
    u8 packed[48]                  # this group's 64 Q6 codes
```

### 7.5 `CONTIGUOUS` (BF16_CTRL / FP32_CTRL)

Raw row-major elements in the stated dtype (bf16 = 2 B, fp32 = 4 B). `group_size=0`,
`scale_dtype=none`. Shape is the canonical tensor-plan shape. For M2.8/M3 TEXT_CORE artifacts,
`linear_attn.conv1d.weight` uses `[10240,4,1]`. Existing raw `[10240,1,4]` q5090 files are legacy
pre-M2.8 artifacts. They must be regenerated before they are used as official M2.8/M3 baseline inputs.

## 8. Padding rules

- Quant matrices pad `N` up to a multiple of 64, and `K` up to 64 (Q4/Q5/Q6) or 128 (W8),
  filling with zeros. `shape` keeps the logical size; `padded_shape` records the tiled size.
- The kernel reads padded tiles branch-free and truncates outputs to `shape` on write-back.
- For Qwen3.6 only the vision `intermediate_size=4304` needs padding: `linear_fc1.weight`
  pads `N` 4304->4352; `linear_fc2.weight` pads `K` 4304->4352. All text/MTP dims are already
  aligned.

## 9. Hashing / integrity

- `name_hash`: FNV-1a-64 over the UTF-8 canonical name. Reference:
  `h = 0xcbf29ce484222325; for b in name: h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF`.
  (Chosen over the policy doc's suggested xxh3_64 to stay dependency-free and trivial in C++.)
- `crc32`: `zlib.crc32` over each tensor's payload bytes.
- `sha256_safetensors_index`: sha256 of the source `model.safetensors.index.json`.

## 10. Canonical names for sliced/reshaped tensors

Fused source tensors are split into separate logical entries (payload granularity == kernel call
granularity), with these canonical names:

```text
model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight  # [10240,5120]
  -> .in_proj_qkv.q   rows [0:2048]     Q4
  -> .in_proj_qkv.k   rows [2048:4096]  Q4
  -> .in_proj_qkv.v   rows [4096:10240] Q5

model.language_model.layers.{L}.self_attn.q_proj.weight         # [12288,5120]
  # per-head INTERLEAVED [query(256)|gate(256)] x24 (view[24,512,5120]); NOT contiguous halves
  -> .q_proj.q        view[24,512,5120][:, :256]  -> [6144,5120]  Q4
  -> .q_proj.gate     view[24,512,5120][:, 256:]  -> [6144,5120]  Q5

model.visual.patch_embed.proj.weight  # Conv3d [1152,3,2,16,16]
  -> reshaped to [1152, 1536]          Q5  (out_channels, in*kt*kh*kw)
```

All other tensors keep their safetensors names.

## 11. Manifest

The converter also writes a sidecar `manifest.json` (human-readable) next to the packed file,
mirroring section 15 of the policy doc: format id, model id, per-segment policy/size, qtypes,
layouts, alignment, effective text bpw, and any `disabled_segments`.
