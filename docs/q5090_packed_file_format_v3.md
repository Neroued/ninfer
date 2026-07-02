# QUS Packed Weight File Format `q5090_w4g64_mixed_v3` (binary spec)

This document is the **decode-optimal canonical binary contract** for Qwen3.6-27B on one RTX 5090. It
is the single weight-file ABI the C++ runtime consumes. There is no other in-tree weight path, no
runtime repack, and **no backward compatibility**: the converter emits v3, the runtime reads v3, and
every consumer (converter, weight store, GEMV and GEMM kernels) targets this format directly with no
transitional or fallback layout.

This document fully specifies the **container format and the code/plane encoding** — a conformant
loader and the converter's byte writer can be built from it alone. The **per-model tensor plan** (the
exact set of blocks, segments, fusion groups, control-tensor shapes, source transforms, and per-tensor
`qtype` / `source_kind` / `source_layer`) is a **normative companion** (see "Normative companions"
below); the two together fully determine a conformant file. The format changes only the **storage order
and grouping**; it never changes the dequantized values (§0 invariants).

### Normative companions

A conformant converter/loader implements this document **plus**:

- **Quantization policy** —
  [qwen3_6_27b_q5090_final_quant_format_v1.md](qwen3_6_27b_q5090_final_quant_format_v1.md): *which
  `qtype` each tensor gets* and the quantizer algorithm whose output this format stores verbatim (§8).
- **TEXT_CORE / VISION tensor plan** —
  [qwen3_6_27b_q5090_v2_tensor_plan.md](qwen3_6_27b_q5090_v2_tensor_plan.md): the TEXT_CORE and
  VISION_ENCODER per-model block/segment/fusion assignment, canonical emission order, control-tensor
  shapes, and source transforms. Its MTP_DRAFT standalone assignment is historical for v2 and is **not**
  normative for v3.
- **MTP_DRAFT v3 assignment** — this document (§10.1, §14, §16) owns the v3 MTP_DRAFT fused
  block/segment/fusion assignment directly: 12 blocks, 16 segments, and two fusion groups. This changes
  only grouping/source transforms inside the existing `q5090_w4g64_mixed_v3` container; it does not
  change `W8G128_F16S`, the file magic, or the version number. The v2 tensor-plan document and the
  quantization-policy document are not edited by this assignment change.

## Why this layout

Single-stream decode (`T == 1`) is **weight-bandwidth bound**: every linear weight is streamed from
HBM once per token, so decode throughput is `BW_eff / bytes_per_token`. The format does **not** reduce
`bytes_per_token` (the codes and scales are unchanged); it maximizes the achievable `BW_eff` and
removes launch/occupancy and unpack waste, by these structural choices:

1. **Row-major split planes.** Each output row's codes are contiguous and all scales live in a separate
   contiguous plane. A GEMV thread/warp streams a row with fully coalesced, vector-width loads — no
   large per-row stride, no shared-memory staging round-trip.
2. **Bit-plane code packing.** Within a quantized block, the low 4 bits of every code are stored as a
   dense **nibble plane**, and any extra high bits (Q5: 1 bit, Q6: 2 bits) are stored in a separate
   dense **high-bit plane**. Each value is reconstructed by `value = sign_extend(low | (high << 4))` —
   pure shift/mask on register-loaded integers, with **no cross-byte bit extraction** and **no
   cross-lane communication**. The nibble plane is structurally identical to a Q4 code plane, so one
   vectorized nibble loader serves Q4, Q5, and Q6.
3. **`N`-as-parallel-axis.** Parallelism is the output dimension `N`, decoupled from any tile width, so
   the CTA count is free. This removes the moderate-`N` occupancy cap of a tiled layout.
4. **First-class fused projection blocks.** Projections that consume the same input activation *at the
   same point in the schedule* are stored as one large-`N` matrix per `qtype`, driven as a shared-input
   dispatch (one GEMV per `qtype`, input reused across them) — mainly to give the small-`N` projections
   enough rows for occupancy.

The same blocks serve prefill GEMM by staging row-range segments into SMEM and dequantizing on chip;
this is a deliberate **decode-first** byte order (see §11). There is exactly one resident packed copy
of every weight; no dequantized BF16/FP32 copy of a quantized weight ever exists on disk or in VRAM.

## 0. Conventions and invariants

- All integers are **little-endian**. Offsets are **absolute** file offsets unless stated otherwise.
- **Multi-byte scalar payloads are little-endian.** FP16 scales and `CONTIGUOUS` BF16/FP32 elements use
  IEEE-754 little-endian byte order (least-significant byte first). BF16 is the high 16 bits of
  IEEE-754 binary32, stored as a little-endian `u16`.
- **Zero-fill.** Every reserved field and **all padding** — index/string-table padding, the gap before
  the 4096-aligned payload region, and every inter-plane / inter-block 256-byte alignment pad — MUST be
  zero. Because `crc32` (§15) covers a block's plane padding, zero-fill is required for deterministic
  CRC. A verifier MUST reject nonzero reserved/padding bytes; a loader SHOULD.
- File starts with a 4096-byte header. Module, tensor, segment, and fusion-group index tables and the
  string table follow. The payload region starts at a 4096-byte boundary; each tensor payload is
  **256-byte aligned**.
- **No baked runtime math transforms**: `A_log`/`dt_bias` upcast to FP32 (bf16 on disk), RMSNorm
  weights stored raw (no `+1`), `A = -exp(A_log)` remains the runtime's job.
- **Canonical TEXT_CORE conv1d layout** is `[10240,4,1]` (`[conv_dim,gdn_conv_k,1]`).

**Normative invariants (each independently verifiable):**

1. **Value preservation.** For every quantized tensor, the stored FP16 scales and signed integer codes
   equal, value-for-value, the quantization policy's output for that tensor. Dequantizing a tensor
   yields exactly the policy's numbers. The format chooses storage order freely and **never** alters a
   value; it is not a requantization and never changes the numeric format.
2. **Single resident copy.** Each weight exists once, in packed form. The format defines no duplicated
   or derived second copy of any weight.
3. **On-chip dequant only.** The format is consumed by streaming packed bytes; dequantization happens
   in registers (GEMV) or SMEM before MMA (GEMM). The format never implies a dequantized copy of a
   quantized weight.
4. **One layout family serves both regimes.** The same block serves decode GEMV (row streaming) and
   prefill GEMM (SMEM-staged Tensor Core); there is no phase-specific repack and no second layout.
5. **Self-describing.** Every block's plane sizes and logical composition are fully determined by its
   `TensorEntry` and its `SegmentRecord`s (with the per-model identity fixed by the tensor-plan
   companion). There is no layout whose field values are left implicit.

## 1. File structure

```text
+--------------------------------------------------+ 0
| FileHeader (4096 bytes)                          |
+--------------------------------------------------+ header_size (4096) = module_index_offset
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

Index regions are **adjacent in this exact order** with no gaps: `module_index_offset == 4096`;
`tensor_index_offset == module_index_offset + module_count*64`;
`segment_index_offset == tensor_index_offset + tensor_count*128`;
`fusion_group_index_offset == segment_index_offset + segment_count*32`;
`string_table_offset == fusion_group_index_offset + fusion_group_count*64`;
`payload_offset == align_up(string_table_offset + string_table_bytes, 4096)`. Modules are contiguous
ranges of the tensor index in the order TEXT_CORE, then MTP_DRAFT, then VISION_ENCODER. The segment and
fusion-group tables are global (indexed by blocks). All three module kinds are first-class; which appear
in a given file is a conversion choice. The current Qwen3.6-27B text-decode artifact contains only the
TEXT_CORE module; the per-module tensor plans are in the tensor-plan companion (§7, §14).

## 2. FileHeader (4096 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 16 | char | `magic` = `"Q5090MIXEDV3\0\0\0\0"` |
| 16 | 4 | u32 | `version` = 3 |
| 20 | 4 | u32 | `endian` = `0x01020304` |
| 24 | 4 | u32 | `header_size` = 4096 |
| 28 | 4 | u32 | `tensor_count` (number of blocks) |
| 32 | 4 | u32 | `module_count` (1..3) |
| 36 | 4 | u32 | `layer_count` = 64 |
| 40 | 4 | u32 | `flags` (see below) |
| 44 | 4 | u32 | `segment_count` |
| 48 | 8 | u64 | `module_index_offset` |
| 56 | 8 | u64 | `module_index_bytes` |
| 64 | 8 | u64 | `tensor_index_offset` |
| 72 | 8 | u64 | `tensor_index_bytes` |
| 80 | 8 | u64 | `string_table_offset` |
| 88 | 8 | u64 | `string_table_bytes` |
| 96 | 8 | u64 | `payload_offset` |
| 104 | 8 | u64 | `payload_bytes` (sum of all block payload sizes incl. inter-block pad) |
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

**`flags` semantics.** bit0 `TEXT_PRESENT`, bit1 `MTP_PRESENT`, bit2 `VISION_PRESENT` — each set iff the
corresponding `module_kind` appears in the module table; bit0 is always set, and the bits MUST agree
with the module table (the current artifact sets only bit0). bit3 `CALIBRATED` — set iff any tensor's
codes/scales came from a calibrated quantizer pass (§8). Bits 4..31 are reserved and MUST be zero. A
loader MUST reject a file whose reserved flag bits are set or whose module-present bits disagree with
the module table.

## 3. ModuleRecord (64 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `module_kind` (0=TEXT_CORE, 1=MTP_DRAFT, 2=VISION_ENCODER) |
| 4 | 4 | u32 | `module_version` = 3 |
| 8 | 8 | u64 | `tensor_index_begin` (first block) |
| 16 | 8 | u64 | `tensor_index_count` (block count) |
| 24 | 8 | u64 | `payload_offset` |
| 32 | 8 | u64 | `payload_bytes` |
| 40 | 4 | u32 | `load_policy` (0=RESIDENT, 1=LAZY_GPU, 2=CPU_PINNED_THEN_GPU) |
| 44 | 4 | u32 | `flags` (reserved; MUST be zero — no module-level flags are defined) |
| 48 | 16 | u8 | reserved zero |

Default `load_policy`: TEXT=RESIDENT, MTP=RESIDENT, VISION=LAZY_GPU. A loader MUST reject a nonzero
module `flags`. The modules' `[tensor_index_begin, tensor_index_begin+tensor_index_count)` ranges
partition `[0, tensor_count)` contiguously in TEXT→MTP→VISION order.

## 4. TensorEntry (128 bytes) — one block

A **block** is the atomic stored unit: a single matrix with one `qtype`, one `layout`, `K` input
columns, and `N` output rows. A block holds one or more logical projections as contiguous row-ranges
(see §5). For a quantized block (`ROW_SPLIT`) the payload is a nibble plane, an optional high-bit
plane, and a scale plane, each covering all `N` rows (§9.2).

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
| 40 | 16 | u32[4] | `padded_shape` (`[N, K_pad, 1, 1]`, `K_pad = align_up(K,128)`; §9.2) |
| 56 | 4 | u32 | `group_size` (64 for Q4/Q5/Q6, 128 for W8, 0 for CONTIGUOUS) |
| 60 | 2 | u16 | `scale_dtype` (0=none, 1=FP16) |
| 62 | 2 | u16 | `segment_count` (≥1) |
| 64 | 8 | u64 | `payload_offset` (absolute, 256-aligned) |
| 72 | 8 | u64 | `payload_bytes` (block payload **size** = `scale_rel + scale_plane_bytes`; §9.2) |
| 80 | 4 | u32 | `source_layer` (0..63, or 0xFFFFFFFF for globals) |
| 84 | 4 | u32 | `source_kind` (block-level identity; see rule below) |
| 88 | 4 | u32 | `crc32` (zlib CRC-32 over the whole payload) |
| 92 | 4 | u32 | `segment_begin` (index of this block's first `SegmentRecord`) |
| 96 | 2 | u16 | `fusion_group_id` (0=NONE; §7) |
| 98 | 2 | u16 | `fusion_index` (position of this block within its fusion group; 0 if NONE) |
| 100 | 8 | u64 | `nibble_plane_bytes` (low-4-bit base plane; for W8 the full int8 plane; §9.2) |
| 108 | 8 | u64 | `high_plane_bytes` (high-bit plane; 0 for Q4 / W8 / CONTIGUOUS) |
| 116 | 8 | u64 | `scale_plane_bytes` (0 for CONTIGUOUS) |
| 124 | 4 | u8 | reserved zero |

`nibble_plane_bytes`, `high_plane_bytes`, and `scale_plane_bytes` are always present and always equal
the values computed in §9.2; `payload_bytes` is the block payload **size** (relative span), not an
absolute end. The runtime **must** assert these. The block's logical projections are the
`segment_count` `SegmentRecord`s at `[segment_begin, segment_begin + segment_count)`.

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
the canonical name in `name_*`. A segment's plane sub-ranges are the matching row-ranges of all three
planes — nibble at `nibble_off + row_begin*G*nib`, high at `high_off + row_begin*G*hi`, scale at
`scale_off + row_begin*G*2` (`nib`/`hi`/`G` per §9.2; addressing per §9.3). Segments of one block
partition `[0, N)` exactly and in increasing `row_begin` order. For a standalone block the single
segment's `name_*` equals the block's `name_*`; for a fused block each segment carries its own
projection name while the block carries the group name.

For a `CONTIGUOUS` block (not an `[N,K]` matrix — e.g. rank-1 norms `[5120]`/`[256]`/`[48]` or rank-3
`conv1d [10240,4,1]`), there is exactly one segment with `row_begin = 0` and `row_count = shape[0]`
(the leading logical dim), and `high_plane_bytes = scale_plane_bytes = 0`. The `[0, N)` row-partition
and output-row semantics in this section apply to `ROW_SPLIT` blocks only.

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

Values `4..63` are reserved for future fusion groups (including MTP/VISION fusion).

`source_kind` (u32) — the complete enum across all modules. TEXT_CORE kinds:

| value | name | value | name | value | name |
|---:|---|---:|---|---:|---|
| 0 | `OTHER` (fused-block placeholder) | 13 | `GDN_IN_PROJ_A` | 32 | `ATTN_K` |
| 1 | `EMBED` | 14 | `GDN_IN_PROJ_B` | 33 | `ATTN_V` |
| 2 | `LM_HEAD` | 15 | `GDN_IN_PROJ_Q` | 34 | `ATTN_Q_NORM` |
| 3 | `FINAL_NORM` | 16 | `GDN_IN_PROJ_K` | 35 | `ATTN_K_NORM` |
| 4 | `INPUT_LAYERNORM` | 17 | `GDN_IN_PROJ_V` | 36 | `ATTN_O` |
| 5 | `POST_ATTN_LAYERNORM` | 18 | `GDN_IN_PROJ_Z` | 40 | `MLP_GATE` |
| 10 | `GDN_A_LOG` | 19 | `GDN_NORM` | 41 | `MLP_UP` |
| 11 | `GDN_DT_BIAS` | 20 | `GDN_OUT_PROJ` | 42 | `MLP_DOWN` |
| 12 | `GDN_CONV1D` | 30 | `ATTN_Q` | 31 | `ATTN_GATE` |

`MTP_DRAFT` (`module_kind=1`) adds the kinds below; its attention / MLP / norm linears **reuse** the
shared `ATTN_*` / `MLP_*` / `INPUT_LAYERNORM` / `POST_ATTN_LAYERNORM` kinds above, disambiguated by
`module_kind` + `source_layer`:

| value | name | value | name |
|---:|---|---:|---|
| 50 | `MTP_FC` | 52 | `MTP_PRE_FC_NORM_HID` |
| 51 | `MTP_PRE_FC_NORM_EMB` | 53 | `MTP_NORM` |

`VISION_ENCODER` (`module_kind=2`):

| value | name | value | name | value | name |
|---:|---|---:|---|---:|---|
| 60 | `VIS_PATCH_EMBED` | 67 | `VIS_BLOCK_FC1` | 74 | `VIS_BLOCK_NORM2_B` |
| 61 | `VIS_PATCH_EMBED_BIAS` | 68 | `VIS_BLOCK_FC1_BIAS` | 75 | `VIS_MERGER_FC1` |
| 62 | `VIS_POS_EMBED` | 69 | `VIS_BLOCK_FC2` | 76 | `VIS_MERGER_FC1_BIAS` |
| 63 | `VIS_BLOCK_QKV` | 70 | `VIS_BLOCK_FC2_BIAS` | 77 | `VIS_MERGER_FC2` |
| 64 | `VIS_BLOCK_QKV_BIAS` | 71 | `VIS_BLOCK_NORM1_W` | 78 | `VIS_MERGER_FC2_BIAS` |
| 65 | `VIS_BLOCK_PROJ` | 72 | `VIS_BLOCK_NORM1_B` | 79 | `VIS_MERGER_NORM_W` |
| 66 | `VIS_BLOCK_PROJ_BIAS` | 73 | `VIS_BLOCK_NORM2_W` | 80 | `VIS_MERGER_NORM_B` |

Values `6..9`, `21..29`, `37..39`, `43..49`, `54..59`, and `81..95` are reserved (unused). A loader MUST
reject a block whose `source_kind` is not a defined value for its `module_kind` (or `OTHER` for a fused
block). The TEXT_CORE assignment is in §10 and the tensor-plan companion; the v3 MTP_DRAFT assignment
is in §10.1 / §14 / §16; the VISION_ENCODER assignment remains in the tensor-plan companion §6.

## 8. Quantization values (weight-only, symmetric, no zero point)

The format stores, **verbatim**, the quantization policy's per-group output; it never recomputes scales
or codes (invariant 0.1). For each group of `group_size` consecutive K-elements of one row the policy
produces an fp16 scale `scale16` and `group_size` signed integer codes `q_i`, with dequant
`w_i = scale16 * q_i`. The policy's **default** quantizer is per-group symmetric max-abs:

```text
qmax    = 2^(bits-1) - 1           # Q4=7, Q5=15, Q6=31, W8=127
scale   = max(abs(group)) / qmax   # fp32
scale16 = fp16(scale)              # stored AND used to quantize
q_i     = clamp(round_half_even(w_i / scale16), qmin, qmax)
```

`qmin = -(qmax+1)` for Q4/Q5/Q6; `qmin = -127` for W8. All-zero group: `scale = 0`, codes `0`. A
nonzero group whose scale underflows fp16 is bumped to the smallest positive fp16 subnormal.

When the policy runs a **calibrated** pass, it MAY choose `scale16` per group by a different criterion
(e.g. error-minimizing search) while keeping the same per-group, symmetric, fp16-scale, no-zero-point
structure; the converter then sets header `flags.CALIBRATED`. Either way the stored `scale16`/`q_i` are
the policy's final output and the format reproduces them exactly. The format defines the *container* for
these values, not the algorithm that chose them.

## 9. Payload encodings

### 9.1 Code packing (bit-plane)

A signed code `q_i` is stored as its `bits`-wide two's-complement pattern
`u_i = q_i & ((1 << bits) - 1)`. `u_i` is split into a **low nibble** and the **high bits**:

```text
low_i  = u_i & 0x0F                 # bits 0..3
high_i = u_i >> 4                   # Q5: 1 bit (0..1); Q6: 2 bits (0..3); Q4: none; W8: see below
```

Per group of `group_size` codes, the format emits:

- **Nibble run** (the low-4-bit base plane) — `group_size / 2` bytes. Byte `b` (`0 ≤ b < group_size/2`)
  holds `low_{2b} | (low_{2b+1} << 4)`. For **Q4** this is the whole code.
- **High run** (the high-bit plane) — present only for Q5/Q6:
  - **Q5**: 1 high bit per code → `group_size / 8` bytes (8 for `group_size=64`). The high bits are
    packed LSB-first: high bit of code `c` is at bit `c & 7` of byte `c >> 3`.
  - **Q6**: 2 high bits per code → `group_size / 4` bytes (16 for `group_size=64`). Packed LSB-first,
    two consecutive bits per code: bits `2c` and `2c+1` of the run are `high_c & 1` and
    `(high_c >> 1) & 1` (i.e. bits 4 and 5 of `u_c`). Reconstruct `high_c = (run >> (2c)) & 0x3`.
- **W8** has **no nibble split and no high run**: the base ("nibble") plane stores one signed `int8`
  per code, `group_size` (= 128) bytes per group; `high_i` is unused.

Per-group byte counts:

| qtype | bits | base ("nibble") bytes/group | high bytes/group | total bytes/group |
|---|---:|---:|---:|---:|
| Q4 | 4 | 32 | 0 | 32 |
| Q5 | 5 | 32 | 8 | 40 |
| Q6 | 6 | 32 | 16 | 48 |
| W8 | 8 | 128 (one `int8` per code) | 0 | 128 |

**Decode (reconstruction):**

```text
# Q4/Q5/Q6:
u_c = low_c | (high_c << 4)         # Q5: 5-bit; Q6: 6-bit; Q4: u_c = low_c
q_c = sign_extend(u_c, bits)        # Q4 from bit3, Q5 from bit4, Q6 from bit5
# W8:
q_c = (int8) base_byte_c            # already signed; no merge, no sign-extend step
# all qtypes:
w_c = scale16(group) * q_c
```

Worked Q5 example: policy code `q = -5`. `u = -5 & 0x1F = 27 = 0b1_1011`. `low = 0b1011 = 11`,
`high = 0b1 = 1`. Stored: nibble `11` in the nibble run, bit set in the high run. Decode:
`u = 11 | (1 << 4) = 27`; `sign_extend(27, 5) = 27 - 32 = -5`. ✔

### 9.2 `ROW_SPLIT` (every quantized block)

Logical `[N, K]`. `K` is padded to **`K_pad = align_up(K, 128)`** — a multiple of both `128` and
`group_size`. This is the single mandatory K-alignment rule: it makes `G = K_pad / group_size` **even**
for the group-64 qtypes (and integral for W8's group-128), which guarantees every plane's per-row run
is 16-byte aligned (`nib·G`, Q5 high `8·G`, Q6 high `16·G`, W8 base `128·G` are all multiples of 16,
since `G` even ⇒ `8·G` is a multiple of 16). `N` is **not** padded. Let `nib` = base bytes/group from
§9.1 (`32` for Q4/Q5/Q6, `128` for W8) and `hi` = high bytes/group (`0` Q4/W8, `8` Q5, `16` Q6). The
payload is **three row-major planes** over all `N` rows (Q4/W8: two — the high plane is empty), each
starting 256-aligned:

```text
# nibble (base) plane: low-4-bit (or int8 for W8) codes, row-major
for n in 0 .. N-1:
  for g in 0 .. G-1:
    u8 nibble[nib]          # (row n, group g) base run, per §9.1

# 256-byte alignment padding (zero)

# high plane: high-bit codes, row-major (absent when hi == 0)
for n in 0 .. N-1:
  for g in 0 .. G-1:
    u8 high[hi]             # (row n, group g) high run, per §9.1

# 256-byte alignment padding (zero; absent when hi == 0)

# scale plane: row-major, each row's scales contiguous
for n in 0 .. N-1:
  for g in 0 .. G-1:
    fp16 scale              # scale of (row n, group g), little-endian
```

Sizes and **relative** plane offsets (always equal to the `TensorEntry` fields):

```text
nibble_plane_bytes = N * G * nib
high_plane_bytes   = N * G * hi                          # 0 for Q4 / W8
scale_plane_bytes  = N * G * 2

# offsets RELATIVE to the block's payload_offset:
nibble_rel = 0
high_rel   = align_up(nibble_plane_bytes, 256)
scale_rel  = high_rel + align_up(high_plane_bytes, 256)  # == high_rel when hi == 0
payload_bytes = scale_rel + scale_plane_bytes            # block payload SIZE (not an absolute end)
```

The absolute device/file addresses used by consumers are `nibble_off = payload_offset + nibble_rel`,
`high_off = payload_offset + high_rel` (unused when `hi == 0`), `scale_off = payload_offset + scale_rel`
(§9.3).

**ABI guarantees (byte facts):**

- Row `n`'s base codes are one contiguous run of `G*nib` bytes at `nibble_off + n*G*nib`; its high bits
  one contiguous run of `G*hi` bytes at `high_off + n*G*hi`; its scales one contiguous run of `G*2`
  bytes at `scale_off + n*G*2`.
- The nibble plane of a Q5/Q6 block is **structurally identical to a Q4 code plane**: the same
  vector-width nibble loader serves Q4, Q5, and Q6 nibble planes.
- Each plane begins 256-aligned, and (because of the `K_pad = align_up(K,128)` rule) each row's run
  inside a plane begins 16-byte aligned, so a 16-byte vector load of any plane is legal.
- The high plane is one uninterrupted dense byte stream. A code's high bit(s) sit at a **fixed bit
  position within a single byte** (Q5: bit `c&7` of byte `c>>3`; Q6: bits `(2c)&7..` of byte
  `(2c)>>3`), so reconstruction is a local shift/mask — never a cross-byte funnel shift, never
  cross-lane communication.
- For a fused block, the rows of distinct projections are adjacent row-ranges within each of the three
  planes — the block is one genuine `[N, K]` matrix, not a concatenation of separate plane sets.

Representative blocks (`nibble` / `high` / `scale` plane bytes):

| block | `[N,K]` | qtype | `G` | nibble plane | high plane | scale plane | segments |
|---|---:|---|---:|---:|---:|---:|---|
| `mlp.down` | `[5120,17408]` | Q5 | 272 | 44,564,480 | 11,141,120 | 2,785,280 | `MLP_DOWN[0,5120)` |
| `o_proj` | `[5120,6144]` | Q5 | 96 | 15,728,640 | 3,932,160 | 983,040 | `ATTN_O[0,5120)` |
| `gdn.out_proj` | `[5120,6144]` | Q5 | 96 | 15,728,640 | 3,932,160 | 983,040 | `GDN_OUT_PROJ[0,5120)` |
| `gdn.in_z` | `[6144,5120]` | Q5 | 80 | 15,728,640 | 3,932,160 | 983,040 | `GDN_IN_PROJ_Z[0,6144)` |
| `lm_head` | `[248320,5120]` | Q6 | 80 | 635,699,200 | 317,849,600 | 39,731,200 | `LM_HEAD[0,248320)` |
| `embed_tokens` | `[248320,5120]` | Q6 | 80 | 635,699,200 | 317,849,600 | 39,731,200 | `EMBED[0,248320)` |
| `ATTN_IN` Q4 block | `[7168,5120]` | Q4 | 80 | 18,350,080 | 0 | 1,146,880 | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)` |
| `ATTN_IN` Q5 block | `[7168,5120]` | Q5 | 80 | 18,350,080 | 4,587,520 | 1,146,880 | `ATTN_GATE[0,6144)`, `ATTN_V[6144,7168)` |
| `GDN_IN` Q4 block | `[4096,5120]` | Q4 | 80 | 10,485,760 | 0 | 655,360 | `GDN_IN_PROJ_Q[0,2048)`, `GDN_IN_PROJ_K[2048,4096)` |
| `GDN_IN` Q5 block | `[6144,5120]` | Q5 | 80 | 15,728,640 | 3,932,160 | 983,040 | `GDN_IN_PROJ_V[0,6144)` |
| `MLP_GATEUP` block | `[34816,5120]` | Q4 | 80 | 89,128,960 | 0 | 5,570,560 | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` |

Total code bytes per block (`nibble + high`) equal `N*G*(nib+hi)` — `40·N·G` Q5, `48·N·G` Q6,
`32·N·G` Q4 — i.e. `bytes_per_token` is unchanged by the plane split.

### 9.3 Plane addressing (runtime view contract)

The loader copies each block's payload (the `payload_bytes` span starting at `payload_offset`) to a
256-aligned device base `block_base`, and exposes the block to GEMV/GEMM kernels as a **plane
descriptor**:

```text
struct BlockView {
  u16   qtype, layout;
  i32   N, K, K_pad, G, group_size;   // G = K_pad/group_size
  i32   nib, hi;                       // base/high bytes per group (§9.1)
  void* nibble_ptr = block_base + 0;
  void* high_ptr   = (hi != 0) ? block_base + align_up(nibble_plane_bytes,256) : nullptr;
  void* scale_ptr  = block_base + scale_rel;   // fp16
};
```

A **segment / projection view** of rows `[r0, r0+rc)` shares the block's three plane pointers; row `r`
of the view (output row `r - r0`) reads:

```text
base codes : (u8*)nibble_ptr + r*G*nib       # G*nib bytes
high bits  : (u8*)high_ptr   + r*G*hi         # G*hi bytes  (only if hi != 0)
scales     : (fp16*)scale_ptr + r*G           # G fp16
```

`high_ptr == nullptr` signals a single-plane block (Q4/W8); a consumer of such a block uses only the
base plane. These pointer/offset rules — not the kernel internals — are the ABI a conformant GEMV/GEMM
relies on; they are fully determined by the `TensorEntry` plane-byte fields and §9.2.

### 9.4 `CONTIGUOUS` (BF16_CTRL / FP32_CTRL)

Raw row-major elements in the stated dtype, little-endian. `group_size=0`, `scale_dtype=none`,
`nibble_plane_bytes = payload_bytes`, `high_plane_bytes = 0`, `scale_plane_bytes = 0`, one segment
spanning the whole tensor. Used by norms, GDN `A_log`/`dt_bias`, conv1d, and the `[48,5120]` GDN
`in_a`/`in_b` dense-control gates.

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
  `in_v` are computed up front.
- `GDN_IN_PROJ_A` / `GDN_IN_PROJ_B` (the `[48,5120]` dense-control gates) are **not** group members;
  they are standalone `CONTIGUOUS` blocks.
- Standalone (no group, `fusion_group_id = NONE`): `ATTN_O`, `GDN_OUT_PROJ`, `GDN_IN_PROJ_Z`,
  `MLP_DOWN`, `LM_HEAD`, `EMBED`, and all control tensors.

The exact per-layer block/segment/fusion counts and emission order are in the tensor-plan companion.

### 10.1 Fused projection groups (MTP_DRAFT)

MTP_DRAFT reuses the existing module-agnostic fusion ids. Its member blocks are distinguished from
TEXT_CORE by `TensorEntry.module_kind = MTP_DRAFT`; no new fusion id is introduced.

| group | source_layer | member blocks (`qtype`, `[N,K]`) | segments (row-ranges) | `total_n` |
|---|---:|---|---|---:|
| `ATTN_IN` | 0 | W8 `[14336,5120]` named `mtp.layers.0.attn_in.w8` | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)`, `ATTN_GATE[7168,13312)`, `ATTN_V[13312,14336)` | 14336 |
| `MLP_GATEUP` | 0 | W8 `[34816,5120]` named `mtp.layers.0.mlp.gateup.w8` | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` | 34816 |

Both fused MTP blocks have block-level `source_kind = OTHER`; segment identity is authoritative. The
MTP raw `self_attn.q_proj.weight [12288,5120]` is interleaved per head as
`[query(256) | gate(256)] x 24`, so the `ATTN_Q` and `ATTN_GATE` segments MUST use the same
value-defining de-interleave transforms as TEXT_CORE full-attention q_proj. A contiguous
`[0:6144] / [6144:12288]` split is invalid.

## 11. Consumption model (normative intent)

This section fixes how the format is meant to be executed so the byte layout is unambiguous. It does
not specify kernel internals; the addressing ABI is §9.3.

- A `ROW_SPLIT` block is one GEMV over its `N` rows: outputs for rows `[0, N)` are produced from the
  block's nibble/high/scale planes and the block's input vector. Per-row output is written to the
  logical projection by segment (`row_begin..row_begin+row_count` → `source_kind`). The consumer reads
  each row's base run (vector-width), its high run (Q5/Q6), and its scales, and reconstructs each code
  as in §9.1 before the multiply-accumulate.
- A fusion group is executed as one GEMV per member block (i.e. per `qtype`), each over its block's
  rows, scattering each segment's rows to its logical projection output. The shared input is read once
  per member block (not once per projection) and is reused across the group's blocks via L2; the format
  enables this shared-input dispatch but does not mandate a single physical input load.
- Prefill GEMM consumes the same blocks by staging `[row-range × k-range]` segments into SMEM
  (coalesced per row from the nibble and high planes), dequantizing on chip, and feeding Tensor Cores.
  No alternate or duplicated layout is produced for prefill.

The L1/L2 surfaces required for the above (a segment/group-aware GEMV path and the fused-projection
call site) are the assumed execution target; the format does not preserve any single-weight-per-call
fallback.

## 12. Padding and alignment

- **K padding.** Quantized blocks pad `K` to `K_pad = align_up(K, 128)` (§9.2), filling the padded
  K-range with zero codes (and, for the extra groups, zero high bits and a zero scale per §8). `shape`
  keeps the logical `K`; `padded_shape` records `[N, K_pad]`. Padded codes contribute nothing. This
  single rule supersedes any per-qtype K-alignment; there is no "odd-G" special case.
- **N padding.** `ROW_SPLIT` does **not** pad `N`. `shape[0] == padded_shape[0] == N`. Output-row tail
  handling is a kernel concern, not a storage concern.
- **Plane alignment.** Relative to the block `payload_offset`: nibble plane at `0`; high plane at
  `align_up(nibble_plane_bytes, 256)`; scale plane at `high_rel + align_up(high_plane_bytes, 256)`.
  When `high_plane_bytes == 0` (Q4/W8) the scale plane immediately follows the aligned nibble plane.
  All inter-plane pad bytes are zero (§0).
- **Block / member alignment.** Each block payload is 256-aligned. Fusion-group member blocks are
  emitted consecutively, each 256-aligned; `FusionGroupRecord.payload_bytes` covers the inter-member
  (zero) padding.

## 13. Structural validation (loader / verifier conformance)

A conformant loader MUST reject a file that fails any of the **structural** checks 1–7 (a failure there
can mis-read weights). Checks marked *(verifier)* — 8 `name_hash`, 9 zero-fill, 10 crc/dequant — are
**integrity** checks for the offline auditor; a loader SHOULD apply them but MAY skip them at load time.
The reserved-**flag**-bit rejection in check 1 is structural (loader MUST). All checks are computable
from this document alone.

1. **Header.** `magic == "Q5090MIXEDV3\0\0\0\0"`; `version == 3`; `endian == 0x01020304`;
   `header_size == 4096`; `format_minor` is a known value (`0`); `module_count ∈ [1,3]`;
   `layer_count == 64`. `flags` reserved bits (4..31) are zero and the present-bits agree with the
   module table (§2).
2. **Index adjacency / sizes.** The six index/string offsets satisfy the adjacency chain of §1, and
   `module_index_bytes == module_count*64`, `tensor_index_bytes == tensor_count*128`,
   `segment_index_bytes == segment_count*32`, `fusion_group_index_bytes == fusion_group_count*64`.
   `payload_offset` is 4096-aligned and equals `align_up(string_table_offset + string_table_bytes,
   4096)`.
3. **Modules.** `module_kind`s are distinct and ordered TEXT→MTP→VISION; the
   `[tensor_index_begin, +tensor_index_count)` ranges partition `[0, tensor_count)` contiguously; each
   module's `flags == 0`; each module's `payload_offset/payload_bytes` span covers exactly its blocks'
   payloads.
4. **Payload region.** Each block `payload_offset` is 256-aligned and lies in
   `[file payload_offset, file_size)`; blocks appear in **tensor-index order** with no overlap; block
   `payload_offset + payload_bytes ≤ next block payload_offset` (or `file_size` for the last); the
   header `payload_bytes` and `file_size` are consistent with the last block's end.
5. **Per-block fields.** `group_size`/`scale_dtype` match `qtype` (Q4/Q5/Q6: 64/FP16; W8: 128/FP16;
   CONTIGUOUS: 0/none). Then, **by layout**:
   - **ROW_SPLIT**: `padded_shape[0] == shape[0] == N`, `padded_shape[1] == align_up(shape[1],128)`,
     `shape[2]==shape[3]==1`; `nibble_plane_bytes`/`high_plane_bytes`/`scale_plane_bytes` equal the §9.2
     formulas; `payload_bytes == scale_rel + scale_plane_bytes`.
   - **CONTIGUOUS**: `padded_shape == shape` (all four dims, no K-padding); `high_plane_bytes == 0`,
     `scale_plane_bytes == 0`, `nibble_plane_bytes == payload_bytes`.
6. **Segments.** `[segment_begin, segment_begin+segment_count) ⊆ [0, segment_count_total)`; a block's
   segments partition `[0, N)` exactly with strictly increasing `row_begin` and no gap/overlap; if
   `segment_count == 1`, block `source_kind/source_layer == segment[0]`'s; if `segment_count > 1`,
   block `source_kind == OTHER (0)`. Each `source_kind` is a defined value for its module (§7).
7. **Fusion groups.** Members are `block_count` consecutive blocks at `first_block_tensor_index`,
   consecutive in payload; each member's `fusion_group_id == group_id`, equal `source_layer`, equal
   `K`, `fusion_index == position`; `total_n == Σ member N`; `shared_k == K`.
8. **String table.** Every `name_offset + name_len < string_table_bytes`, the byte at
   `name_offset + name_len` is `NUL`, and (*verifier*) `name_hash == FNV-1a-64(name)` (§15).
9. **(verifier) Zero-fill.** All reserved struct fields and all index/string-table/inter-plane/
   inter-block padding bytes are zero (§0). (Reserved *flag* bits are structural — see check 1.)
10. **(verifier)** `crc32` matches the block payload (§15); dequantizing each block reproduces the
    policy values bit-for-bit (invariant 0.1).

## 14. Layout assignment policy

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
| MTP `fc`, `o_proj`, `down_proj` | W8 | `ROW_SPLIT` (group 128) | standalone |
| MTP `attn q/k/gate/v` | W8 | `ROW_SPLIT` (group 128) | `ATTN_IN` (one block, four segments) |
| MTP MLP `gate`, `up` | W8 | `ROW_SPLIT` (group 128) | `MLP_GATEUP` (one block, two segments) |
| vision quantized linears | Q4/Q5/W8 | `ROW_SPLIT` | standalone (vision biases/norms `CONTIGUOUS`) |

Every quantized tensor in every module uses `ROW_SPLIT`; every dense control tensor uses `CONTIGUOUS`.
There is no tiled layout anywhere. The class table above fixes layout/grouping. TEXT_CORE and
VISION_ENCODER exact per-tensor assignment is in the tensor-plan companion (Normative companions);
MTP_DRAFT exact v3 assignment is the following canonical block order:

| block name | qtype/layout | shape | source kind / segments |
|---|---|---:|---|
| `mtp.fc.weight` | W8 row-split | `[5120,10240]` | `MTP_FC`, `NO_LAYER` |
| `mtp.pre_fc_norm_embedding.weight` | BF16 contiguous | `[5120]` | `MTP_PRE_FC_NORM_EMB`, `NO_LAYER` |
| `mtp.pre_fc_norm_hidden.weight` | BF16 contiguous | `[5120]` | `MTP_PRE_FC_NORM_HID`, `NO_LAYER` |
| `mtp.layers.0.input_layernorm.weight` | BF16 contiguous | `[5120]` | `INPUT_LAYERNORM`, layer 0 |
| `mtp.layers.0.attn_in.w8` | W8 row-split | `[14336,5120]` | block `OTHER`; segments `ATTN_Q/K/GATE/V`, layer 0 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 contiguous | `[256]` | `ATTN_Q_NORM`, layer 0 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 contiguous | `[256]` | `ATTN_K_NORM`, layer 0 |
| `mtp.layers.0.self_attn.o_proj.weight` | W8 row-split | `[5120,6144]` | `ATTN_O`, layer 0 |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 contiguous | `[5120]` | `POST_ATTN_LAYERNORM`, layer 0 |
| `mtp.layers.0.mlp.gateup.w8` | W8 row-split | `[34816,5120]` | block `OTHER`; segments `MLP_GATE/MLP_UP`, layer 0 |
| `mtp.layers.0.mlp.down_proj.weight` | W8 row-split | `[5120,17408]` | `MLP_DOWN`, layer 0 |
| `mtp.norm.weight` | BF16 contiguous | `[5120]` | `MTP_NORM`, `NO_LAYER` |

A converter or verifier implements these assignments to produce/validate the concrete
block/segment/fusion set. A full artifact contains 1164 blocks, 1312 segments, and 130 fusion groups:
TEXT_CORE contributes 819/963/128, MTP_DRAFT contributes 12/16/2, and VISION_ENCODER contributes
333/333/0.

`embed_tokens` is gather-driven (one row per token), not GEMV-streamed, so the row/plane split gives it
no throughput benefit and costs a couple of extra small transactions per lookup (a row's nibble run,
high run, and scales are in three planes rather than inline). It uses `ROW_SPLIT` anyway as a
deliberate uniformity choice: the cost is negligible (a few extra bytes of latency on a ~4 KB lookup,
dwarfed by per-token weight streaming), it keeps a single quantized layout, and it lets `embed` and
`lm_head` (identical `[248320,5120] Q6`) share one layout.

## 15. Hashing / integrity

- `name_hash`: FNV-1a-64 over the UTF-8 canonical name, for both `TensorEntry` (block name) and
  `SegmentRecord` (projection name)
  (`h=0xcbf29ce484222325; for b: h=((h^b)*0x100000001b3) & 2^64-1`).
- `crc32`: `zlib.crc32` over each block's whole payload (nibble plane + pad + high plane + pad + scale
  plane, or the raw bytes for `CONTIGUOUS`), with all pad bytes zero (§0). This is an offline
  converter/auditor integrity field; the cpp runtime loader does not recompute it during normal load.
- `sha256_safetensors_index`: sha256 of the source `model.safetensors.index.json` (32 raw bytes in the
  header; the manifest carries the lowercase hex form).

A converter or auditor MAY verify correctness by dequantizing each block and asserting bit-identical
results against the quantization policy reference (invariant 0.1), independent of byte order.

## 16. Canonical names for sliced/reshaped source tensors

Source-model tensors that map to multiple projections, or multiple source tensors that map into one
fused block, are reconciled by the segment table; each `SegmentRecord` carries the projection's
canonical `name`/`name_hash` (§5). The full per-tensor name/transform list is in the tensor-plan
companion; representative cases:

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

mtp.layers.0.self_attn.q_proj.weight [12288,5120]   # interleaved [query(256)|gate(256)] x24
  -> ATTN_Q   [6144,5120] W8 (MTP ATTN_IN block segment)
  -> ATTN_GATE[6144,5120] W8 (MTP ATTN_IN block segment)
mtp.layers.0.self_attn.k_proj.weight -> ATTN_K [1024,5120] W8 (MTP ATTN_IN block segment)
mtp.layers.0.self_attn.v_proj.weight -> ATTN_V [1024,5120] W8 (MTP ATTN_IN block segment)

mtp.layers.0.mlp.gate_proj.weight -> MLP_GATE [17408,5120] W8 (MTP MLP_GATEUP block segment)
mtp.layers.0.mlp.up_proj.weight   -> MLP_UP   [17408,5120] W8 (MTP MLP_GATEUP block segment)
```

Each block's `name`/`name_hash` is the converter-assigned canonical name (for a standalone block, the
projection name; for a fused block, the group name, e.g. `layers.{L}.attn_in.q4`); per-projection names
are recovered from segment `name_*`.

## 17. Manifest

The sidecar `manifest.json` (file name `<weights>.manifest.json`, next to the `.qus`) is an
informational mirror of the file the converter emits; the binary header is authoritative. Required
fields:

```json
{
  "format": "q5090_w4g64_mixed_v3",
  "format_version": 3,
  "format_minor": 0,
  "binary_spec": "docs/q5090_packed_file_format_v3.md",
  "tensor_plan": "docs/q5090_packed_file_format_v3.md",
  "value_source": "qwen3_6_27b_q5090_final_quant_format_v1 (policy)",
  "weights_file": "qwen3_6_27b.q5090_w4g64_mixed_v3.qus",
  "file_bytes": 0,
  "sha256_safetensors_index": "<64 hex chars>",
  "calibrated": false,
  "alignment": { "header": 4096, "payload": 4096, "block": 256, "k_pad": 128, "group_size": 64 },
  "layouts": ["ROW_SPLIT", "CONTIGUOUS"],
  "code_planes": ["nibble", "high", "scale"],
  "qtypes": ["Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G128_F16S", "BF16_CTRL", "FP32_CTRL"],
  "modules": ["TEXT_CORE", "MTP_DRAFT", "VISION_ENCODER"],
  "absent_modules": [],
  "tensor_count": 1164,
  "segment_count": 1312,
  "fusion_group_count": 130,
  "fusion_groups": [
    {"module": "TEXT_CORE", "group_id": "ATTN_IN", "group_count": 16, "blocks_per_group": 2, "total_n": 14336, "shared_k": 5120},
    {"module": "TEXT_CORE", "group_id": "GDN_IN", "group_count": 48, "blocks_per_group": 2, "total_n": 10240, "shared_k": 5120},
    {"module": "TEXT_CORE", "group_id": "MLP_GATEUP", "group_count": 64, "blocks_per_group": 1, "total_n": 34816, "shared_k": 5120},
    {"module": "MTP_DRAFT", "group_id": "ATTN_IN", "group_count": 1, "blocks_per_group": 1, "total_n": 14336, "shared_k": 5120},
    {"module": "MTP_DRAFT", "group_id": "MLP_GATEUP", "group_count": 1, "blocks_per_group": 1, "total_n": 34816, "shared_k": 5120}
  ],
  "effective_text_bpw": 4.8716
}
```

`tensor_count`/`segment_count`/`fusion_group_count` and `file_bytes` are filled by the converter and
MUST equal the header values (a block is one `TensorEntry`, so `tensor_count` is the block count — what
the tensor-plan companion calls `block_count`); `modules` lists the modules present in this file and
`absent_modules` the supported-but-omitted ones; `effective_text_bpw` is the policy value. The manifest
is not consumed by the runtime loader (which reads only the binary header) but MUST be consistent with
it.

## 18. Non-goals and scope

- **No bytes/token reduction.** The format stores the same codes/scales; the gain is achievable
  `BW_eff` and reduced launch/occupancy/unpack waste, not less traffic.
- **No requantization, no numeric-format change.** No FP4/FP6/MXFP/NVFP; values equal the policy.
- **No second resident copy and no dequant-to-BF16/FP32** of any quantized weight.
- **Per-group fp16 scales.** One fp16 scale per `(row, group)` of 64 (128 for W8). The format does not
  use hierarchical/sub-block quantized scales.
- **Float-accumulation consumption assumed.** The byte layout targets a dequantize-to-float GEMV/GEMM;
  it neither requires nor encodes an integer-dot (quantized-activation) execution path.
- **Decode-first prefill, explicitly.** The global byte order is GEMV-optimal; prefill GEMM consumes it
  via SMEM-staged per-row segment loads. Because prefill is compute-bound and lower priority, this is a
  deliberate choice, not a regression; the format does not store a GEMM-optimal duplicate.
- **No legacy layout, no fallback, no migration path.** Only `ROW_SPLIT` + `CONTIGUOUS` exist.
- **MTP_DRAFT / VISION_ENCODER are first-class but selectively loadable.** The format, the `source_kind`
  enum (§7), and the converter support all three modules in the full artifact. Runtime load policy may
  keep MTP/VISION unloaded until requested, but their block metadata is still present. VISION remains
  standalone per the tensor-plan companion; MTP_DRAFT uses the fused v3 assignment in this document.

## 19. Differences from `q5090_w4g64_mixed_v2`

This format supersedes v2; there is no dual-format path. The **only** substantive on-disk change is the
code packing; the validation/addressing/flags clarifications below tighten the contract without
changing v2-shared bytes.

| aspect | v2 | v3 |
|---|---|---|
| magic / `version` / `format` | `Q5090MIXEDV2` / 2 / `..._v2` | `Q5090MIXEDV3` / 3 / `..._v3` |
| Q5/Q6 code packing | one group bitstream, codes concatenated LSB-first; decode needs cross-byte funnel-shift | **bit-plane**: low-4-bit nibble plane + separate high-bit plane (Q5: 1 bit, Q6: 2 bits); decode `sign_extend(low\|high<<4)`, shift/mask only |
| ROW_SPLIT planes | 2 (code + scale) | 3 (nibble + high [empty for Q4/W8] + scale), each 256-aligned |
| `TensorEntry` plane fields | `code_plane_bytes` (100), `scale_plane_bytes` (108) | `nibble_plane_bytes` (100), `high_plane_bytes` (108), `scale_plane_bytes` (116) |
| `payload_bytes` definition | (ambiguous) | block payload **size** = `scale_rel + scale_plane_bytes` (relative span) |
| K padding | to `group_size` (with an odd-`G` Q5→128 caveat) | single rule `K_pad = align_up(K,128)`; no special case |
| Q4 / W8 packing, scales, values, bytes/token | nibble / int8; per-group fp16; policy values | **unchanged** (bit-identical dequant; same total code bytes per block) |
| `source_kind` MTP/VISION | spec listed placeholders `50..53 MTP_*`, `60..80 VIS_*` | full enum listed in §7; **values unchanged** (match v1 + `tensor.h`/`qtypes.py`) |
| MTP_DRAFT assignment | standalone W8 linears in the v2 tensor-plan companion | v3-owned fused assignment: 12 blocks, 16 segments, `ATTN_IN` + `MLP_GATEUP` fusion groups; no version bump inside v3 |
| module support | TEXT/MTP/VISION first-class | **unchanged** (the full converter artifact carries all three; runtime load policy may skip optional modules) |
| plane addressing, structural validation, flags, manifest, FP endianness | implicit | **made normative** here (§9.3, §13, §2/§3, §17, §0) |

Everything not listed is identical to v2 in structure and meaning. v3 is a pure relayout of the Q5/Q6
code bits: same dequantized values (invariant 0.1), same `bytes_per_token`, designed so a warp-per-row
GEMV reconstructs each value with register-only shift/mask and reuses the Q4 nibble loader for the
dominant (low-bit) traffic.
