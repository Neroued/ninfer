# QUS Packed Weight File Format `q5090_w4g64_mixed_v2` (binary spec)

This document is the **decode-optimal canonical binary contract** for Qwen3.6-27B on one RTX 5090. It
is the single weight-file ABI the C++ runtime consumes. There is no other in-tree weight path, no
runtime repack, and **no backward compatibility**: the converter emits v2, the runtime reads v2, and
every consumer (converter, weight store, GEMV and GEMM kernels) targets this format directly with no
transitional or fallback layout.

Read it together with the quantization *policy*
([qwen3_6_27b_q5090_final_quant_format_v1.md](qwen3_6_27b_q5090_final_quant_format_v1.md)) and the
linear backend constraints ([m3-linear-backend-framework.md](m3-linear-backend-framework.md)). The
policy document decides *which qtype each tensor gets*; this document decides *how those values are
laid out on disk*. v2 changes only the **storage order and grouping**; it never changes the
dequantized values (§0 invariants).

## Why v2 exists

Single-stream decode (`T == 1`) is **weight-bandwidth bound**: every linear weight is streamed from
HBM once per token, so decode throughput is `BW_eff / bytes_per_token`. v2 does **not** reduce
`bytes_per_token` (the codes and scales are unchanged); it maximizes the achievable `BW_eff` and
removes launch/occupancy waste, by three structural choices:

1. **Row-major split planes.** Each output row's codes are one contiguous run, and all scales live in a
   second contiguous plane. A GEMV thread/warp streams a row with fully coalesced, vector-width loads —
   no large per-row stride, no shared-memory staging round-trip.
2. **`N`-as-parallel-axis.** Parallelism is the output dimension `N`, decoupled from any tile width, so
   the CTA count is free. This removes the moderate-`N` occupancy cap of a tiled layout.
3. **First-class fused projection blocks.** Projections that consume the same input activation *at the
   same point in the schedule* are stored as one large-`N` matrix per `qtype`, so they are driven as a
   shared-input dispatch (one GEMV per `qtype`, input reused across them) — mainly to give the small-`N`
   projections enough rows for occupancy.

The same blocks serve prefill GEMM by staging row-range segments into SMEM and dequantizing on chip;
this is a deliberate **decode-first** byte order (see §11, §17). There is exactly one resident packed
copy of every weight; no dequantized BF16/FP32 copy of a quantized weight ever exists on disk or in
VRAM (framework §14.2 and decisions 7–8).

## 0. Conventions and invariants

- All integers are **little-endian**. Reserved bytes are **zero**. Offsets are **absolute** file
  offsets unless stated otherwise.
- File starts with a 4096-byte header. Module, tensor, segment, and fusion-group index tables and the
  string table follow. The payload region starts at a 4096-byte boundary; each tensor payload is
  **256-byte aligned**.
- **No baked runtime math transforms**: `A_log`/`dt_bias` upcast to FP32 (bf16 on disk), RMSNorm
  weights stored raw (no `+1`), `A = -exp(A_log)` remains the runtime's job.
- **Canonical TEXT_CORE conv1d layout** is `[10240,4,1]` (`[conv_dim,gdn_conv_k,1]`).

**Normative invariants (each independently verifiable):**

1. **Value preservation.** For every quantized tensor, the stored FP16 scales and signed integer codes
   equal, value-for-value, the quantization policy's output for that tensor. Dequantizing a v2 tensor
   yields exactly the policy's numbers. v2 chooses storage order freely and **never** alters a value;
   it is not a requantization and never changes the numeric format.
2. **Single resident copy.** Each weight exists once, in packed form. The format defines no duplicated
   or derived second copy of any weight.
3. **On-chip dequant only.** The format is consumed by streaming packed bytes; dequantization happens
   in registers (GEMV) or SMEM before MMA (GEMM). The format never implies a dequantized copy of a
   quantized weight.
4. **One layout family serves both regimes.** The same block serves decode GEMV (row streaming) and
   prefill GEMM (SMEM-staged Tensor Core); there is no phase-specific repack and no second layout.
5. **Self-describing.** Every block's plane sizes and logical composition are fully determined by its
   `TensorEntry` and its `SegmentRecord`s. There is no layout whose field values are left implicit.

## 1. File structure

```text
+--------------------------------------------------+ 0
| FileHeader (4096 bytes)                          |
+--------------------------------------------------+ header_size (4096)
| ModuleRecord[module_count]        (64 bytes each)|
+--------------------------------------------------+ tensor_index_offset
| TensorEntry[tensor_count]        (128 bytes each)|   (each TensorEntry is one block)
+--------------------------------------------------+ segment_index_offset
| SegmentRecord[segment_count]      (32 bytes each)|
+--------------------------------------------------+ fusion_group_index_offset
| FusionGroupRecord[fusion_group_count] (64 B each)|
+--------------------------------------------------+ string_table_offset
| String table (NUL-terminated tensor/segment names)|
+--------------------------------------------------+ payload_offset (4096-aligned)
| Block payloads (each 256-aligned, no overlap)    |
+--------------------------------------------------+ file_size
```

Modules are contiguous ranges of the tensor index: TEXT_CORE first, then MTP_DRAFT, then
VISION_ENCODER. The segment and fusion-group tables are global (indexed by blocks).

## 2. FileHeader (4096 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 16 | char | `magic` = `"Q5090MIXEDV2\0\0\0\0"` |
| 16 | 4 | u32 | `version` = 2 |
| 20 | 4 | u32 | `endian` = `0x01020304` |
| 24 | 4 | u32 | `header_size` = 4096 |
| 28 | 4 | u32 | `tensor_count` (number of blocks) |
| 32 | 4 | u32 | `module_count` (1..3) |
| 36 | 4 | u32 | `layer_count` = 64 |
| 40 | 4 | u32 | `flags` (bit0 TEXT, bit1 MTP, bit2 VISION, bit3 calibrated) |
| 44 | 4 | u32 | `segment_count` |
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
| 164 | 4 | u32 | `fusion_group_count` |
| 168 | 32 | u8 | `sha256_safetensors_index` |
| 200 | 8 | u64 | `segment_index_offset` |
| 208 | 8 | u64 | `segment_index_bytes` |
| 216 | 8 | u64 | `fusion_group_index_offset` |
| 224 | 8 | u64 | `fusion_group_index_bytes` |
| 232 | 4 | u32 | `format_minor` = 0 |
| 236 | 3860 | u8 | reserved zero |

## 3. ModuleRecord (64 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `module_kind` (0=TEXT_CORE, 1=MTP_DRAFT, 2=VISION_ENCODER) |
| 4 | 4 | u32 | `module_version` = 2 |
| 8 | 8 | u64 | `tensor_index_begin` (first block) |
| 16 | 8 | u64 | `tensor_index_count` (block count) |
| 24 | 8 | u64 | `payload_offset` |
| 32 | 8 | u64 | `payload_bytes` |
| 40 | 4 | u32 | `load_policy` (0=RESIDENT, 1=LAZY_GPU, 2=CPU_PINNED_THEN_GPU) |
| 44 | 4 | u32 | `flags` |
| 48 | 16 | u8 | reserved zero |

Default `load_policy`: TEXT=RESIDENT, MTP=RESIDENT, VISION=LAZY_GPU.

## 4. TensorEntry (128 bytes) — one block

A **block** is the atomic stored unit: a single matrix with one `qtype`, one `layout`, `K` input
columns, and `N` output rows. A block holds one or more logical projections as contiguous row-ranges
(see §5). For a quantized block (`ROW_SPLIT`) the payload is one code plane followed by one scale plane
covering all `N` rows.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `name_offset` (into string table) |
| 4 | 4 | u32 | `name_len` (excluding NUL) |
| 8 | 8 | u64 | `name_hash` (FNV-1a-64 of canonical block name) |
| 16 | 2 | u16 | `qtype` |
| 18 | 2 | u16 | `layout` (0=ROW_SPLIT, 1=CONTIGUOUS) |
| 20 | 2 | u16 | `module_kind` |
| 22 | 2 | u16 | `ndim` |
| 24 | 16 | u32[4] | `shape` (logical `[N, K, 1, 1]`; `N` = total rows of the block) |
| 40 | 16 | u32[4] | `padded_shape` (`[N, K_pad, 1, 1]`) |
| 56 | 4 | u32 | `group_size` (64 for Q4/Q5/Q6, 128 for W8, 0 for CONTIGUOUS) |
| 60 | 2 | u16 | `scale_dtype` (0=none, 1=FP16) |
| 62 | 2 | u16 | `segment_count` (≥1) |
| 64 | 8 | u64 | `payload_offset` (absolute, 256-aligned) |
| 72 | 8 | u64 | `payload_bytes` (code plane + alignment pad + scale plane) |
| 80 | 4 | u32 | `source_layer` (0..63, or 0xFFFFFFFF for globals) |
| 84 | 4 | u32 | `source_kind` (block-level identity; see rule below) |
| 88 | 4 | u32 | `crc32` (zlib CRC-32 over the whole payload) |
| 92 | 4 | u32 | `segment_begin` (index of this block's first `SegmentRecord`) |
| 96 | 2 | u16 | `fusion_group_id` (0=NONE; §7) |
| 98 | 2 | u16 | `fusion_index` (position of this block within its fusion group; 0 if NONE) |
| 100 | 8 | u64 | `code_plane_bytes` |
| 108 | 8 | u64 | `scale_plane_bytes` (0 for CONTIGUOUS) |
| 116 | 12 | u8 | reserved zero |

`code_plane_bytes` and `scale_plane_bytes` are always present and always equal the values computed in
§9. The runtime **must** assert this. The block's logical projections are the `segment_count`
`SegmentRecord`s at `[segment_begin, segment_begin + segment_count)`.

**`source_kind` rule (no ambiguity for fused blocks).** When `segment_count == 1`, `source_kind` /
`source_layer` mirror the single segment's identity. When `segment_count > 1` (a fused block),
`source_kind` **must** be `OTHER` (0) and the per-projection identities live only in the segments. The
runtime never uses block-level `source_kind` to look up a fused projection; segment identity is
authoritative in all cases. `name`/`name_hash` of a fused block is the converter-assigned group name.

## 5. SegmentRecord (32 bytes)

A segment maps a contiguous row-range of a block to one logical projection and carries that
projection's canonical name and hash, so per-projection identity and naming survive fusion. Standalone
weight = one segment spanning `[0, N)`; a fused block has one segment per fused projection.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `source_kind` (the logical projection identity; §7) |
| 4 | 4 | u32 | `source_layer` (0..63, or 0xFFFFFFFF) |
| 8 | 4 | u32 | `row_begin` (within the owning block) |
| 12 | 4 | u32 | `row_count` (logical output rows of this projection) |
| 16 | 4 | u32 | `name_offset` (canonical projection name, into string table) |
| 20 | 4 | u32 | `name_len` (excluding NUL) |
| 24 | 8 | u64 | `name_hash` (FNV-1a-64 of the canonical projection name) |

The runtime builds its `LayerWeights` pointer table from segments: each segment yields a weight view =
rows `[row_begin, row_begin + row_count)` of its block, with identity `(source_kind, source_layer)` and
the canonical name in `name_*`. Segments of one block partition `[0, N)` exactly and in increasing
`row_begin` order. For a standalone block the single segment's `name_*` equals the block's `name_*`;
for a fused block each segment carries its own projection name while the block carries the group name.

For a `CONTIGUOUS` block (not an `[N,K]` matrix — e.g. rank-1 norms `[5120]`/`[256]`/`[48]` or rank-3
`conv1d [10240,4,1]`), there is exactly one segment with `row_begin = 0` and `row_count = shape[0]`
(the leading logical dim), and `scale_plane_bytes = 0`. The `[0, N)` row-partition and output-row
semantics in this section apply to `ROW_SPLIT` blocks only.

## 6. FusionGroupRecord (64 bytes)

A fusion group binds the blocks that consume the **same input activation** (same `K`) so the runtime
can dispatch them together against one shared input. A group has one member block per `qtype` present
among its projections; a single-`qtype` group is one block.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `group_id` (1=ATTN_IN, 2=GDN_IN, 3=MLP_GATEUP; §7) |
| 4 | 4 | u32 | `source_layer` (0..63) |
| 8 | 4 | u32 | `block_count` (member blocks; ≥1) |
| 12 | 4 | u32 | `shared_input_kind` (`source_kind` of the consumed activation; informational) |
| 16 | 8 | u64 | `first_block_tensor_index` |
| 24 | 8 | u64 | `payload_offset` (start of the group's contiguous payload span) |
| 32 | 8 | u64 | `payload_bytes` (span incl. inter-block 256-alignment padding) |
| 40 | 4 | u32 | `total_n` (sum of member block `N`) |
| 44 | 4 | u32 | `shared_k` (common `K` of all members) |
| 48 | 16 | u8 | reserved zero |

**Group invariants:**

- Member blocks are the `block_count` TensorEntries at
  `[first_block_tensor_index, first_block_tensor_index + block_count)`. They are consecutive in both
  the tensor index and the payload region.
- Every member has `fusion_group_id == group_id`, the same `source_layer`, the same `K`, and
  `fusion_index` equal to its position in `[0, block_count)`.
- A group is consumed as a shared-input dispatch over its member blocks — one GEMV per member block
  (i.e. per `qtype`) — **not** necessarily a single physical GEMV over `total_n` rows (see §11).

## 7. Enums

`qtype` (u16):

| value | name | bits | group | signed range |
|---:|---|---:|---:|---|
| 0 | `Q4G64_F16S` | 4 | 64 | [-8, 7] |
| 1 | `Q5G64_F16S` | 5 | 64 | [-16, 15] |
| 2 | `Q6G64_F16S` | 6 | 64 | [-32, 31] |
| 3 | `W8G128_F16S` | 8 | 128 | [-127, 127] |
| 4 | `BF16_CTRL` | 16 | - | bfloat16 |
| 5 | `FP32_CTRL` | 32 | - | float32 |

`layout` (u16) — the complete set; there are no other layouts:

| value | name | used by |
|---:|---|---|
| 0 | `ROW_SPLIT` | every quantized tensor (Q4/Q5/Q6 with `group_size=64`, W8 with `group_size=128`) |
| 1 | `CONTIGUOUS` | dense control tensors (BF16_CTRL / FP32_CTRL) |

`module_kind` (u16): `0=TEXT_CORE`, `1=MTP_DRAFT`, `2=VISION_ENCODER`.

`scale_dtype` (u16): `0=none`, `1=FP16`.

`fusion_group_id` (u16):

| value | name | member projections (`source_kind`) | per |
|---:|---|---|---|
| 0 | `NONE` | — | standalone blocks |
| 1 | `ATTN_IN` | `ATTN_Q`, `ATTN_K`, `ATTN_GATE`, `ATTN_V` | full-attention layer |
| 2 | `GDN_IN` | `GDN_IN_PROJ_Q`, `GDN_IN_PROJ_K`, `GDN_IN_PROJ_V` | GDN layer (`in_z` excluded; see §10) |
| 3 | `MLP_GATEUP` | `MLP_GATE`, `MLP_UP` | every layer |

`source_kind` (u32) — unchanged from the policy enum: `0 OTHER`, `1 EMBED`, `2 LM_HEAD`,
`3 FINAL_NORM`, `4 INPUT_LAYERNORM`, `5 POST_ATTN_LAYERNORM`, `10 GDN_A_LOG`, `11 GDN_DT_BIAS`,
`12 GDN_CONV1D`, `13 GDN_IN_PROJ_A`, `14 GDN_IN_PROJ_B`, `15 GDN_IN_PROJ_Q`, `16 GDN_IN_PROJ_K`,
`17 GDN_IN_PROJ_V`, `18 GDN_IN_PROJ_Z`, `19 GDN_NORM`, `20 GDN_OUT_PROJ`, `30 ATTN_Q`, `31 ATTN_GATE`,
`32 ATTN_K`, `33 ATTN_V`, `34 ATTN_Q_NORM`, `35 ATTN_K_NORM`, `36 ATTN_O`, `40 MLP_GATE`, `41 MLP_UP`,
`42 MLP_DOWN`, `50..53 MTP_*`, `60..80 VIS_*`.

## 8. Quantization math (weight-only, symmetric, no zero point)

For each group of `group_size` consecutive K-elements of one row:

```text
qmax    = 2^(bits-1) - 1           # Q4=7, Q5=15, Q6=31, W8=127
scale   = max(abs(group)) / qmax   # fp32
scale16 = fp16(scale)              # stored AND used to quantize
q_i     = clamp(round_half_even(w_i / scale16), qmin, qmax)
dequant: w_i ~= scale16 * q_i
```

`qmin = -(qmax+1)` for Q4/Q5/Q6; `qmin = -127` for W8. All-zero group: `scale = 0`, codes `0`. A
nonzero group whose scale underflows fp16 is bumped to the smallest positive fp16 subnormal. These
`scale16` and `q_i` values are exactly what v2 stores (invariant 0.1); v2 only relocates them.

## 9. Payload encodings

### 9.1 Code packing

- **Q4/Q5/Q6**: `bits_per_group = ceil(64 * bits / 8)` → **Q4=32, Q5=40, Q6=48** bytes per 64-element
  group. Within a group the 64 signed codes are concatenated **LSB-first** (bit `j` of code `c` at
  stream position `c*bits + j`) and packed into bytes LSB-first
  (`numpy.packbits(..., bitorder="little")`); the decoder sign-extends from bit `bits-1`. Q4 reduces to
  standard nibble packing.
- **W8**: `bits_per_group = 128` bytes per 128-element group — one signed `int8` per code, no
  bit-packing.

`bpr` below denotes `bits_per_group` for the block's `qtype`.

### 9.2 `ROW_SPLIT` (every quantized block)

Logical `[N, K]`. `K` is padded to a multiple of `group_size` (`K_pad`); `N` is **not** padded. Let
`G = K_pad / group_size` and `bpr` as in §9.1. The payload is two row-major planes over all `N` rows of
the block:

```text
# code plane: row-major, each row contiguous
for n in 0 .. N-1:
  for g in 0 .. G-1:
    u8 code[bpr]            # (row n, group g) codes, packed per §9.1

# 256-byte alignment padding

# scale plane: row-major, each row's scales contiguous
for n in 0 .. N-1:
  for g in 0 .. G-1:
    fp16 scale             # scale of (row n, group g)
```

Sizes (always equal to the `TensorEntry` fields):

```text
code_plane_bytes  = N * G * bpr
scale_plane_off   = align_up(code_plane_bytes, 256)        # relative to payload_offset
scale_plane_bytes = N * G * 2
payload_bytes     = scale_plane_off + scale_plane_bytes
```

**ABI guarantees (byte facts):**

- Row `n`'s codes are one contiguous run of `G*bpr` bytes; its scales one contiguous run of `G*2`
  bytes.
- The code plane is one uninterrupted byte stream (scales are not interleaved into it).
- Each row's code run starts at a 16-byte boundary for all in-scope shapes (all `K` are multiples of
  128 ⇒ `G` even ⇒ Q5's `40*G` is a multiple of 16; Q4 `32*G`, Q6 `48*G`, W8 `128*G` are always
  multiples of 16). A 16-byte vector load of the code plane may span a group boundary; the consumer
  applies each code's group scale by code index. No per-group padding is inserted.
- For a fused block, the rows of distinct projections are simply adjacent row-ranges of the same two
  planes — the block is one genuine `[N, K]` matrix, not a concatenation of separate plane pairs.

**What this enables (kernel-level, not byte guarantees):** coalesced warp-per-row (or row-stripe)
streaming with no shared-memory staging and no cross-warp reduction; vector-width code loads; for a
fusion group, shared-input dispatch where the input is read once per member block and reused across the
group's blocks via L2 (a single *physical* input load is **not** guaranteed — a mixed-`qtype` group
runs one GEMV per member block). These are properties a kernel may realize because of the byte facts
above; they are not promised by the format itself.

Representative blocks:

| block | `[N,K]` | qtype | `G` | `bpr` | code plane | scale plane | segments |
|---|---:|---|---:|---:|---:|---:|---|
| `mlp.down` | `[5120,17408]` | Q5 | 272 | 40 | 55,705,600 | 2,785,280 | `MLP_DOWN[0,5120)` |
| `o_proj` | `[5120,6144]` | Q5 | 96 | 40 | 19,660,800 | 983,040 | `ATTN_O[0,5120)` |
| `gdn.out_proj` | `[5120,6144]` | Q5 | 96 | 40 | 19,660,800 | 983,040 | `GDN_OUT_PROJ[0,5120)` |
| `gdn.in_z` | `[6144,5120]` | Q5 | 80 | 40 | 19,660,800 | 983,040 | `GDN_IN_PROJ_Z[0,6144)` |
| `lm_head` | `[248320,5120]` | Q6 | 80 | 48 | 953,548,800 | 39,731,200 | `LM_HEAD[0,248320)` |
| `embed_tokens` | `[248320,5120]` | Q6 | 80 | 48 | 953,548,800 | 39,731,200 | `EMBED[0,248320)` |
| `ATTN_IN` Q4 block | `[7168,5120]` | Q4 | 80 | 32 | 18,350,080 | 1,146,880 | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)` |
| `ATTN_IN` Q5 block | `[7168,5120]` | Q5 | 80 | 40 | 22,937,600 | 1,146,880 | `ATTN_GATE[0,6144)`, `ATTN_V[6144,7168)` |
| `GDN_IN` Q4 block | `[4096,5120]` | Q4 | 80 | 32 | 10,485,760 | 655,360 | `GDN_IN_PROJ_Q[0,2048)`, `GDN_IN_PROJ_K[2048,4096)` |
| `GDN_IN` Q5 block | `[6144,5120]` | Q5 | 80 | 40 | 19,660,800 | 983,040 | `GDN_IN_PROJ_V[0,6144)` |
| `MLP_GATEUP` block | `[34816,5120]` | Q4 | 80 | 32 | 89,128,960 | 5,570,560 | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` |

### 9.3 `CONTIGUOUS` (BF16_CTRL / FP32_CTRL)

Raw row-major elements in the stated dtype. `group_size=0`, `scale_dtype=none`,
`code_plane_bytes = payload_bytes`, `scale_plane_bytes = 0`, one segment spanning the whole tensor.
Used by norms, GDN `A_log`/`dt_bias`, conv1d, and the `[48,5120]` GDN `in_a`/`in_b` dense-control
gates.

## 10. Fused projection groups (TEXT-core)

Fusion is realized in storage: a fusion group's projections are stored as one block per `qtype`, each a
single `ROW_SPLIT` matrix whose rows are partitioned into the member projections by segments. Member
blocks of a group are emitted consecutively (tensor index and payload).

| group | layers | member blocks (`qtype`, `[N,K]`) | segments (row-ranges) | `total_n` |
|---|---|---|---|---:|
| `ATTN_IN` | 16 full | Q4 `[7168,5120]`; Q5 `[7168,5120]` | Q4: `ATTN_Q[0,6144)`,`ATTN_K[6144,7168)`; Q5: `ATTN_GATE[0,6144)`,`ATTN_V[6144,7168)` | 14336 |
| `GDN_IN` | 48 GDN | Q4 `[4096,5120]`; Q5 `[6144,5120]` | Q4: `GDN_IN_PROJ_Q[0,2048)`,`GDN_IN_PROJ_K[2048,4096)`; Q5: `GDN_IN_PROJ_V[0,6144)` | 10240 |
| `MLP_GATEUP` | 64 all | Q4 `[34816,5120]` | `MLP_GATE[0,17408)`,`MLP_UP[17408,34816)` | 34816 |

Notes:

- A group has one member block per distinct `qtype` among its projections. `ATTN_IN` and `GDN_IN` span
  two qtypes (two blocks each); `MLP_GATEUP` is one qtype (one block, two segments). Each member block
  is itself a clean large-`N` `ROW_SPLIT` matrix.
- **A group fuses only projections computed at the same point in the layer schedule.**
  `GDN_IN_PROJ_Z` is **excluded** from `GDN_IN` and stored standalone: in the GDN mixer `z` is computed
  *late* (after the conv/gating/recurrence) and feeds only the output-gate norm, while `in_q`/`in_k`/
  `in_v` are computed up front. Fusing `z` would force it to be computed early and keep a `[6144,T]`
  activation live across the recurrence during prefill, for no decode benefit (`in_v` at 6144 rows is
  already well occupied). `GDN_IN` therefore covers only the early inputs `in_q`/`in_k`/`in_v`.
- `GDN_IN_PROJ_A` / `GDN_IN_PROJ_B` (the `[48,5120]` dense-control gates) are **not** group members;
  they are standalone `CONTIGUOUS` blocks.
- Standalone (no group, `fusion_group_id = NONE`): `ATTN_O`, `GDN_OUT_PROJ`, `GDN_IN_PROJ_Z`,
  `MLP_DOWN`, `LM_HEAD`, `EMBED`, and all control tensors.

## 11. Consumption model (normative intent)

This section fixes how the format is meant to be executed so the byte layout is unambiguous. It does
not specify kernel internals.

- A `ROW_SPLIT` block is one GEMV over its `N` rows: outputs for rows `[0, N)` are produced from the
  block's code/scale planes and the block's input vector. Per-row output is written to the logical
  projection by segment (`row_begin..row_begin+row_count` → `source_kind`).
- A fusion group is executed as one GEMV per member block (i.e. per `qtype`), each over its block's
  rows, scattering each segment's rows to its logical projection output. The shared input is read once
  per member block (not once per projection) and is reused across the group's blocks via L2; the format
  enables this shared-input dispatch but does not mandate a single physical input load.
- Prefill GEMM consumes the same blocks by staging `[row-range × k-range]` segments into SMEM
  (coalesced per row), dequantizing on chip, and feeding Tensor Cores. No alternate or duplicated
  layout is produced for prefill.

The L1/L2 surfaces required for the above (a segment/group-aware GEMV path and the fused-projection
call site) are the assumed execution target; the format does not preserve any single-weight-per-call
fallback.

## 12. Padding and alignment

- **K padding.** Quantized blocks pad `K` to a multiple of `group_size` (64 for Q4/Q5/Q6, 128 for W8),
  filling with zeros. `shape` keeps the logical `K`; `padded_shape` records `[N, K_pad]`. Padded codes
  are zero and contribute nothing.
- **N padding.** `ROW_SPLIT` does **not** pad `N`. `shape[0] == padded_shape[0] == N`. Output-row tail
  handling is a kernel concern, not a storage concern.
- **Plane alignment.** The code plane starts at the 256-aligned `payload_offset`; the scale plane at
  `payload_offset + align_up(code_plane_bytes, 256)`. Row code-runs are 16-byte aligned for all in-scope
  shapes (§9.2); a future shape with odd `G` under Q5 must pad `K` to a multiple of 128 to preserve it.
- **Block / member alignment.** Each block payload is 256-aligned. Fusion-group member blocks are
  emitted consecutively, each 256-aligned; `FusionGroupRecord.payload_bytes` covers the inter-member
  padding.

## 13. Layout assignment policy

| tensor class | qtype | layout | grouping |
|---|---|---|---|
| attn `q`, `k`, `gate`, `v` | Q4/Q4/Q5/Q5 | `ROW_SPLIT` | `ATTN_IN` (Q4 block: q,k; Q5 block: gate,v) |
| attn `o_proj` | Q5 | `ROW_SPLIT` | standalone |
| GDN `in_q`, `in_k`, `in_v` | Q4/Q4/Q5 | `ROW_SPLIT` | `GDN_IN` (Q4 block: in_q,in_k; Q5 block: in_v) |
| GDN `in_z` | Q5 | `ROW_SPLIT` | standalone (computed late; see §10) |
| GDN `out_proj` | Q5 | `ROW_SPLIT` | standalone |
| GDN `in_a`, `in_b` | BF16_CTRL | `CONTIGUOUS` | standalone |
| MLP `gate`, `up` | Q4 | `ROW_SPLIT` | `MLP_GATEUP` (one block, two segments) |
| MLP `down` | Q5 | `ROW_SPLIT` | standalone |
| `lm_head` | Q6 | `ROW_SPLIT` | standalone |
| `embed_tokens` | Q6 | `ROW_SPLIT` | standalone (consumed by gather; see note) |
| norms, `A_log`, `dt_bias`, conv1d | BF16/FP32 | `CONTIGUOUS` | standalone |
| MTP quantized linears | W8 | `ROW_SPLIT` (group 128) | per MTP fusion if applicable |
| vision quantized linears | Q4/Q5/W8 | `ROW_SPLIT` | standalone (vision biases/norms `CONTIGUOUS`) |

Every quantized tensor in every module uses `ROW_SPLIT`; every dense control tensor uses `CONTIGUOUS`.
There is no tiled layout anywhere. The exact per-tensor `qtype` assignment is owned by the policy doc.

`embed_tokens` is gather-driven (one row per token), not GEMV-streamed, so `ROW_SPLIT`'s row/plane
split gives it no throughput benefit and costs one extra small transaction per lookup (the row's scales
live in the scale plane rather than inline). It uses `ROW_SPLIT` anyway as a deliberate
uniformity-over-gather-locality choice: the cost is negligible (a few extra bytes of latency on a
~4 KB lookup, dwarfed by per-token weight streaming), it keeps a single quantized layout, and it lets
`embed` and `lm_head` (identical `[248320,5120] Q6`) share one layout.

## 14. Hashing / integrity

- `name_hash`: FNV-1a-64 over the UTF-8 canonical name, for both `TensorEntry` (block name) and
  `SegmentRecord` (projection name)
  (`h=0xcbf29ce484222325; for b: h=((h^b)*0x100000001b3) & 2^64-1`).
- `crc32`: `zlib.crc32` over each block's whole payload (code plane + alignment pad + scale plane, or
  the raw bytes for `CONTIGUOUS`).
- `sha256_safetensors_index`: sha256 of the source `model.safetensors.index.json`.

A converter or auditor MAY verify correctness by dequantizing each block and asserting bit-identical
results against the quantization policy reference (invariant 0.1), independent of byte order.

## 15. Canonical names for sliced/reshaped source tensors

Source-model tensors that map to multiple projections, or multiple source tensors that map into one
fused block, are reconciled by the segment table; each `SegmentRecord` carries the projection's
canonical `name`/`name_hash` (§5), so per-projection identity and naming survive fusion:

```text
self_attn.q_proj.weight [12288,5120]   # interleaved [query(256)|gate(256)] x24
  -> ATTN_Q   [6144,5120] Q4  (ATTN_IN Q4 block segment)
  -> ATTN_GATE[6144,5120] Q5  (ATTN_IN Q5 block segment)
self_attn.k_proj.weight -> ATTN_K [1024,5120] Q4 (ATTN_IN Q4 block segment)
self_attn.v_proj.weight -> ATTN_V [1024,5120] Q5 (ATTN_IN Q5 block segment)

linear_attn.in_proj_qkv.weight [10240,5120]
  -> GDN_IN_PROJ_Q [2048,5120] Q4 (GDN_IN Q4 block segment)
  -> GDN_IN_PROJ_K [2048,5120] Q4 (GDN_IN Q4 block segment)
  -> GDN_IN_PROJ_V [6144,5120] Q5 (GDN_IN Q5 block segment)
linear_attn.in_proj_z.weight  -> GDN_IN_PROJ_Z [6144,5120] Q5 (standalone block; computed late)

mlp.gate_proj.weight -> MLP_GATE [17408,5120] Q4 (MLP_GATEUP block segment)
mlp.up_proj.weight   -> MLP_UP   [17408,5120] Q4 (MLP_GATEUP block segment)

model.visual.patch_embed.proj.weight (Conv3d) -> [1152,1536] Q5 ROW_SPLIT (standalone)
```

Each block's `name`/`name_hash` is the converter-assigned canonical name (for a standalone block, the
projection name; for a fused block, the group name, e.g. `layers.{L}.attn.in.q4`); per-projection
names are recovered from segment `name_*`.

## 16. Manifest

The sidecar `manifest.json` mirrors the policy doc with v2 fields:

```json
{
  "format": "q5090_w4g64_mixed_v2",
  "format_version": 2,
  "layouts": ["ROW_SPLIT", "CONTIGUOUS"],
  "block_count": 0,
  "segment_count": 0,
  "fusion_groups": [
    {"group_id": "ATTN_IN", "layers": 16, "blocks_per_group": 2, "total_n": 14336},
    {"group_id": "GDN_IN", "layers": 48, "blocks_per_group": 2, "total_n": 10240},
    {"group_id": "MLP_GATEUP", "layers": 64, "blocks_per_group": 1, "total_n": 34816}
  ],
  "value_source": "qwen3_6_27b_q5090_final_quant_format_v1 (policy)",
  "effective_text_bpw": 4.8716
}
```

`effective_text_bpw` is the policy value; v2 stores the same codes and scales, so the TEXT-core payload
size differs from a tiled layout only by the removed `N`-padding and changed inter-tensor alignment
(zero net change for this model, whose `N` are all aligned).

## 17. Design rationale vs the v1 tiled layout (对标)

| Aspect | v1 `TILE_N64_K64` | v2 `ROW_SPLIT` blocks | Why |
|---|---|---|---|
| Per-row K access | scattered across `K64` tiles (`kTileBytes` stride) | one contiguous code run + one contiguous scale run | Coalesced, vectorizable, no SMEM staging for GEMV. |
| Scale placement | inline per `K64` group head | separate row-major scale plane | Uninterrupted, 16-byte-aligned code stream. |
| Parallel axis | `N64` row-stripes (CTA count capped) | free `N`-parallel | Remove moderate-`N` occupancy cap. |
| `N` padding | padded to 64 | unpadded | No wasted rows. |
| Projection storage | one tensor per projection (small-`N`) | fused blocks + segments (large-`N`) | Occupancy for small-`N` projections; shared-input dispatch (L2-reused). |
| Layout count | `TILE_N64_K64` + `TILE_N64_K128` + `ROW_GROUPED_G64` + `CONTIGUOUS` | `ROW_SPLIT` + `CONTIGUOUS` | One quantized layout, uniform across all modules. |
| Prefill GEMM | one contiguous `64×64` tile slab | per-row coalesced segments staged to SMEM | Decode-first byte order; prefill is compute-bound (§18). |
| Dequant values | policy output | identical policy output | Pure relayout. |

This is **not** a migration contract: v1 is removed; the converter, weight store, and all kernels read
v2 only.

## 18. Non-goals and scope

- **No bytes/token reduction.** v2 stores the same codes/scales; the gain is achievable `BW_eff` and
  reduced launch/occupancy waste, not less traffic.
- **No requantization, no numeric-format change.** No FP4/FP6/MXFP/NVFP; values equal the policy.
- **No second resident copy and no dequant-to-BF16/FP32** of any quantized weight.
- **Decode-first prefill, explicitly.** The global byte order is GEMV-optimal; prefill GEMM consumes it
  via SMEM-staged per-row segment loads (per-row coalesced, not one contiguous tile slab). Because
  prefill is compute-bound and lower priority, this is a deliberate choice, not a regression; v2 does
  not store a GEMM-optimal duplicate.
- **No legacy layout, no fallback, no migration path.** `TILE_N64_K64`, `TILE_N64_K128`, and
  `ROW_GROUPED_G64` do not exist in v2.
- **No kernel or runtime implementation** is specified here; this is the byte contract and its intended
  consumption only. Bit-packing for Q5/Q6 retains the policy's LSB-first scheme (§9.1); an
  unpack-friendly repacking is out of scope unless a future profile shows the GEMV is unpack-bound
  rather than DRAM-bound.
