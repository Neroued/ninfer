# QUS Packed Artifact Format `q5090_w4g64_mixed_v4_2` (binary spec)

This document is the **decode-optimal canonical binary contract** for Qwen3.6-27B on one RTX 5090. It
is the single model-artifact ABI the C++ runtime consumes. It contains the packed weights and the
CPU-only tokenizer assets required by the text frontend. There is no other in-tree weight path, no
runtime repack, and **no backward compatibility**: the converter emits v4.2, the runtime reads v4.2, and
every consumer (converter, weight store, GEMV and GEMM kernels) targets this format directly with no
transitional or fallback layout.

This document is **self-contained**: it fully specifies the container format, the code/plane encoding,
and the tensor-assignment policy (§7, §10, §14) needed to build a conformant converter and loader from
this file alone. The exhaustive per-layer emission (which layer emits which block, in which order)
follows mechanically from the assignment policy applied across the 64 decoder layers and is not
re-listed tensor-by-tensor. For weights, the format changes only the **storage order and grouping**;
it never changes the dequantized values (§0 invariants). Tokenizer assets are separate raw bytes.

The format includes an **optional shortlisted draft `lm_head`** for MTP
speculative drafting: a standalone Q4 weights block plus a small `int32` index -> vocab id map. It is
optional and, when present, changes only *which* tokens are proposed during drafting; verification
always uses the full `lm_head`, so emitted tokens are unaffected (§18). The selection method and the
measured decision to ship it are recorded in
[2026-07-06-lm-head-draft-q4-decision.md](2026-07-06-lm-head-draft-q4-decision.md).

v4.2 removes `W8G128_F16S`, assigns tag 3 to the sole W8 format `W8G32_F16S`, moves `I32_CTRL` to tag
6, changes both vision merger FC weights to W8G32, and changes the flattened vision patch embedding
to Q6G64. It retains the self-contained tokenizer and independent `LM_HEAD_DRAFT` module introduced
by the retired v4.1 artifact. No earlier minor is accepted and no compatibility parser exists.

## Why this layout

Single-stream decode (`T == 1`) is **weight-bandwidth bound**: every linear weight is streamed from
HBM once per token, so decode throughput is `BW_eff / bytes_per_token`. For the Q4/Q5/Q6 and W8G32
tensors, the format does **not** reduce `bytes_per_token`; it maximizes achievable `BW_eff` and removes
launch/occupancy and unpack waste. MTP_DRAFT and the vision merger use W8G32 for fine W8 scale
granularity. The
optional draft `lm_head` is the other: it stores a frequency-shortlisted subset of vocab rows at Q4,
so drafting streams far fewer bytes per draft step (§14, §18).

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
- **Multi-byte scalar payloads are little-endian.** FP16 scales and `CONTIGUOUS` BF16/FP32/int32
  elements use IEEE-754 (or two's-complement, for int32) little-endian byte order (least-significant
  byte first). BF16 is the high 16 bits of IEEE-754 binary32, stored as a little-endian `u16`.
- **Zero-fill.** Every reserved field and **all padding** — index/string/tokenizer padding, the gap before
  the 4096-aligned payload region, and every inter-plane / inter-block 256-byte alignment pad — MUST be
  zero. Because `crc32` (§15) covers a block's plane padding, zero-fill is required for deterministic
  CRC. A verifier MUST reject all nonzero reserved/padding bytes. A loader MUST reject them in the
  header/catalog/tokenizer region and MAY skip unselected weight-payload padding.
- File starts with a 4096-byte header. Module, tensor, segment, and fusion-group index tables and the
  string table follow, then the tokenizer index/data region. The weight payload region starts at a
  4096-byte boundary; each tensor payload is **256-byte aligned**.
- **No baked runtime math transforms**: `A_log`/`dt_bias` upcast to FP32 (bf16 on disk), RMSNorm
  weights stored raw (no `+1`), `A = -exp(A_log)` remains the runtime's job.
- **Canonical TEXT_CORE conv1d layout** is `[10240,4,1]` (`[conv_dim,gdn_conv_k,1]`).

**Normative invariants (each independently verifiable):**

1. **Value preservation.** For every quantized tensor, the stored FP16 scales and signed integer codes
   equal, value-for-value, the quantizer's output for that tensor (§8). Dequantizing a tensor yields
   exactly those numbers. The format chooses storage order freely and **never** alters a value; it is
   not a requantization and never changes the numeric format. (The optional draft `lm_head` is the one
   tensor whose values are freshly derived — Q4 of the original bf16 rows — rather than shared with
   another block; the format still stores that quantizer output verbatim. See §8/§14.)
2. **Single resident copy.** Each weight exists once, in packed form. The format defines no duplicated
   or derived second copy of any weight. The draft `lm_head` is a distinct, smaller tensor (a
   shortlisted subset re-quantized at Q4), not a second copy of the full `lm_head`.
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
+--------------------------------------------------+ tokenizer_index_offset (64-aligned)
| TokenizerRecord[3]                 (64 bytes each)|
+--------------------------------------------------+ tokenizer_data_offset (64-aligned)
| tokenizer assets + inter-asset zero padding      |
+--------------------------------------------------+ payload_offset (4096-aligned)
| Block payloads (each 256-aligned, no overlap)    |
+--------------------------------------------------+ file_size
```

Index regions are **adjacent in this exact order** with no gaps: `module_index_offset == 4096`;
`tensor_index_offset == module_index_offset + module_count*64`;
`segment_index_offset == tensor_index_offset + tensor_count*128`;
`fusion_group_index_offset == segment_index_offset + segment_count*32`;
`string_table_offset == fusion_group_index_offset + fusion_group_count*64`;
`tokenizer_index_offset == align_up(string_table_offset + string_table_bytes, 64)`;
`tokenizer_data_offset == align_up(tokenizer_index_offset + 3*64, 64)`; and
`payload_offset == align_up(tokenizer_data_offset + tokenizer_data_bytes, 4096)`. Modules are
contiguous ranges of the tensor index in the canonical order TEXT_CORE, optional LM_HEAD_DRAFT,
optional MTP_DRAFT, optional VISION_ENCODER. This order is semantic and is not numeric enum order.
The segment and fusion-group tables are global (indexed by blocks). All four module kinds are
first-class; which optional modules appear is a conversion choice. The per-module assignment is in
§7, §10, and §14.

## 2. FileHeader (4096 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 16 | char | `magic` = `"Q5090MIXEDV4\0\0\0\0"` |
| 16 | 4 | u32 | `version` = 4 |
| 20 | 4 | u32 | `endian` = `0x01020304` |
| 24 | 4 | u32 | `header_size` = 4096 |
| 28 | 4 | u32 | `tensor_count` (number of blocks) |
| 32 | 4 | u32 | `module_count` (1..4) |
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
| 232 | 4 | u32 | `format_minor` = 2 |
| 236 | 4 | u32 | `tokenizer_record_count` = 3 |
| 240 | 4 | u32 | `tokenizer_record_size` = 64 |
| 244 | 4 | u32 | `tokenizer_flags` = 0 |
| 248 | 8 | u64 | `tokenizer_index_offset` |
| 256 | 8 | u64 | `tokenizer_index_bytes` = 192 |
| 264 | 8 | u64 | `tokenizer_data_offset` |
| 272 | 8 | u64 | `tokenizer_data_bytes` (includes inter-asset padding, excludes final 4096 pad) |
| 280 | 3816 | u8 | reserved zero |

**`flags` semantics.** bit0 `TEXT_PRESENT`, bit1 `MTP_PRESENT`, bit2 `VISION_PRESENT` — each set iff the
corresponding `module_kind` appears in the module table; bit0 is always set, and the bits MUST agree
with the module table. bit3 `CALIBRATED` — set iff any tensor's codes/scales came from a calibrated
quantizer pass (§8). bit4 `LM_HEAD_DRAFT_PRESENT` — set **iff** an `LM_HEAD_DRAFT` ModuleRecord is
present (§14). A full artifact sets
`TEXT_PRESENT | MTP_PRESENT | VISION_PRESENT | LM_HEAD_DRAFT_PRESENT`. Bits 5..31 are reserved and MUST
be zero. A loader MUST reject a file whose reserved flag bits are set or whose presence bits disagree
with the module table. `DRAFT_HEAD_PRESENT` is a retired v4.0 name and is not an alias.

### 2.1 TokenizerRecord (64 bytes)

The artifact contains exactly three CPU-only raw UTF-8 assets. They are not tensors and never enter a
device arena or H2D transfer plan.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `kind` |
| 4 | 4 | u32 | `encoding` = 0 (`RAW_UTF8`) |
| 8 | 8 | u64 | `data_offset` (absolute) |
| 16 | 8 | u64 | `data_bytes` |
| 24 | 4 | u32 | `crc32` over asset bytes only |
| 28 | 4 | u32 | reserved zero |
| 32 | 32 | u8 | `sha256` over asset bytes only |

The records appear exactly once each and in this exact order:

| record | kind | source asset | maximum size |
|---:|---:|---|---:|
| 0 | 1 | `tokenizer.json` (`TOKENIZER_JSON`) | 256 MiB |
| 1 | 2 | `merges.txt` (`MERGES_TXT`) | 64 MiB |
| 2 | 3 | `generation_config.json` (`GENERATION_CONFIG_JSON`) | 1 MiB |

All three assets are non-empty valid UTF-8. The converter copies the source bytes verbatim: it does
not normalize JSON, line endings, or terminal newlines and does not append NUL. The first asset starts
at `tokenizer_data_offset`; each later asset starts at
`align_up(previous.data_offset + previous.data_bytes, 64)`. `tokenizer_data_bytes` spans from
`tokenizer_data_offset` through the final asset byte and includes inter-asset zero padding. All bytes
between the string table and tokenizer index, between tokenizer assets, and between the final asset
and the 4096-aligned weight `payload_offset` MUST be zero.

The SHA-256 field records the digest of the bytes embedded in this artifact; it does not select one
globally frozen tokenizer identity. The converter and offline verifier recompute CRC32 and SHA-256.
The runtime recomputes CRC32, checks UTF-8 and tokenizer-region padding, then lets `QwenTokenizer`
parse the embedded assets when a text frontend needs them. Missing `merges.txt` or
`generation_config.json` is invalid in v4.2; there is no hard-coded stop-token fallback.

The converter validates tokenizer semantics while building the artifact. `model.type` MUST be `BPE`;
`model.vocab` MUST be non-empty and all vocab and
`added_tokens` ids MUST be unique integers in `[0, FileHeader.vocab_size)`. Added-token ids MUST NOT
overlap the base vocab; every required added-token field MUST have the canonical type, and
`single_word`, `lstrip`, `rstrip`, and `normalized` MUST all be false. Every effective `merges.txt`
line is exactly two non-empty symbols separated by one ASCII space and merge pairs MUST be unique.
`eos_token_id` is a non-empty integer or integer array whose ids are present in the embedded
vocab/added-token set.
The TEXT_CORE `EMBED` and full `LM_HEAD` blocks MUST both have logical shape
`[FileHeader.vocab_size, FileHeader.hidden_size]`; this coupling prevents tokenizer ids from addressing
outside the resident embedding table.

`tokenizer_config.json` from the selected converter tokenizer directory (`--tokenizer`, default
`--model`) is an input for draft-shortlist special-token selection but is not a runtime asset because
the current C++ tokenizer does not consume it. Adding another runtime tokenizer
asset requires a new explicit format revision; reserved fields cannot be repurposed silently.

## 3. ModuleRecord (64 bytes)

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `module_kind` (0=TEXT_CORE, 1=MTP_DRAFT, 2=VISION_ENCODER, 3=LM_HEAD_DRAFT) |
| 4 | 4 | u32 | `module_version` = 4 |
| 8 | 8 | u64 | `tensor_index_begin` (first block) |
| 16 | 8 | u64 | `tensor_index_count` (block count) |
| 24 | 8 | u64 | `payload_offset` |
| 32 | 8 | u64 | `payload_bytes` |
| 40 | 4 | u32 | reserved zero (v4.0 `load_policy` is removed) |
| 44 | 4 | u32 | reserved zero |
| 48 | 16 | u8 | reserved zero |

Runtime residency comes only from explicit feature requests; it is never encoded in the artifact. A
loader MUST reject either nonzero u32 at offsets 40/44. The modules'
`[tensor_index_begin, tensor_index_begin+tensor_index_count)` ranges partition `[0, tensor_count)`
contiguously in TEXT->LM_HEAD_DRAFT->MTP->VISION order with absent optional modules omitted.

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
| 56 | 4 | u32 | `group_size` (64 for Q4/Q5/Q6, 32 for W8G32, 0 for CONTIGUOUS incl. I32_CTRL) |
| 60 | 2 | u16 | `scale_dtype` (0=none, 1=FP16) |
| 62 | 2 | u16 | `segment_count` (>=1) |
| 64 | 8 | u64 | `payload_offset` (absolute, 256-aligned) |
| 72 | 8 | u64 | `payload_bytes` (block payload **size** = `scale_rel + scale_plane_bytes`; §9.2) |
| 80 | 4 | u32 | `source_layer` (0..63, or 0xFFFFFFFF for globals) |
| 84 | 4 | u32 | `source_kind` (block-level identity; see rule below) |
| 88 | 4 | u32 | `crc32` (zlib CRC-32 over the whole payload) |
| 92 | 4 | u32 | `segment_begin` (index of this block's first `SegmentRecord`) |
| 96 | 2 | u16 | `fusion_group_id` (0=NONE; §7) |
| 98 | 2 | u16 | `fusion_index` (position of this block within its fusion group; 0 if NONE) |
| 100 | 8 | u64 | `nibble_plane_bytes` (low-4-bit base plane; for W8G32 the full int8 plane; for CONTIGUOUS the whole payload; §9.2/§9.4) |
| 108 | 8 | u64 | `high_plane_bytes` (high-bit plane; 0 for Q4 / W8G32 / CONTIGUOUS) |
| 116 | 8 | u64 | `scale_plane_bytes` (0 for CONTIGUOUS) |
| 124 | 4 | u8 | reserved zero |

`nibble_plane_bytes`, `high_plane_bytes`, and `scale_plane_bytes` are always present and always equal
the values computed in §9.2 (ROW_SPLIT) or §9.4 (CONTIGUOUS); `payload_bytes` is the block payload
**size** (relative span), not an absolute end. The runtime **must** assert these. The block's logical
projections are the `segment_count` `SegmentRecord`s at `[segment_begin, segment_begin + segment_count)`.

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

For a `CONTIGUOUS` block (not an `[N,K]` matrix — e.g. rank-1 norms `[5120]`/`[256]`/`[48]`, the
rank-1 int32 draft-head id-map `[131072]`, or rank-3 `conv1d [10240,4,1]`), there is exactly one
segment with `row_begin = 0` and `row_count = shape[0]` (the leading logical dim), and
`high_plane_bytes = scale_plane_bytes = 0`. The `[0, N)` row-partition and output-row semantics in this
section apply to `ROW_SPLIT` blocks only.

## 6. FusionGroupRecord (64 bytes)

A fusion group binds the blocks that consume the **same input activation** (same `K`) so the runtime
can dispatch them together against one shared input. A group has one member block per `qtype` present
among its projections; a single-`qtype` group is one block.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `group_id` (1=ATTN_IN, 2=GDN_IN, 3=MLP_GATEUP; §7) |
| 4 | 4 | u32 | `source_layer` (0..63) |
| 8 | 4 | u32 | `block_count` (member blocks; >=1) |
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
| 3 | `W8G32_F16S` | 8 | 32 | [-127, 127] |
| 4 | `BF16_CTRL` | 16 | - | bfloat16 |
| 5 | `FP32_CTRL` | 32 | - | float32 |
| 6 | `I32_CTRL` | 32 | - | signed int32 |

`I32_CTRL` is a dense control dtype (raw two's-complement `int32` elements, `CONTIGUOUS`,
`group_size = 0`, `scale_dtype = none`). It carries the draft-head id-map (§9.4, §14).

`layout` (u16) — the complete set; there are no other layouts:

| value | name | used by |
|---:|---|---|
| 0 | `ROW_SPLIT` | every quantized tensor (Q4/Q5/Q6 with `group_size=64`, W8G32 with `group_size=32`) |
| 1 | `CONTIGUOUS` | dense control tensors (BF16_CTRL / FP32_CTRL / I32_CTRL) |

`module_kind` (u16): `0=TEXT_CORE`, `1=MTP_DRAFT`, `2=VISION_ENCODER`,
`3=LM_HEAD_DRAFT`.

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

| value | name | value | name |
|---:|---|---:|---|
| 0 | `OTHER` (fused-block placeholder) | 1 | `EMBED` |
| 2 | `LM_HEAD` | 3 | `FINAL_NORM` |
| 4 | `INPUT_LAYERNORM` | 5 | `POST_ATTN_LAYERNORM` |
| 10 | `GDN_A_LOG` | 11 | `GDN_DT_BIAS` |
| 12 | `GDN_CONV1D` | 13 | `GDN_IN_PROJ_A` |
| 14 | `GDN_IN_PROJ_B` | 15 | `GDN_IN_PROJ_Q` |
| 16 | `GDN_IN_PROJ_K` | 17 | `GDN_IN_PROJ_V` |
| 18 | `GDN_IN_PROJ_Z` | 19 | `GDN_NORM` |
| 20 | `GDN_OUT_PROJ` | 30 | `ATTN_Q` |
| 31 | `ATTN_GATE` | 32 | `ATTN_K` |
| 33 | `ATTN_V` | 34 | `ATTN_Q_NORM` |
| 35 | `ATTN_K_NORM` | 36 | `ATTN_O` |
| 40 | `MLP_GATE` | 41 | `MLP_UP` |
| 42 | `MLP_DOWN` | | |

`LM_HEAD_DRAFT` (6) is the optional shortlisted draft `lm_head` weights block; `LM_HEAD_DRAFT_IDMAP`
(7) is its paired `int32` index -> vocab id map. These two source kinds are valid only when
`module_kind=LM_HEAD_DRAFT`; both use `source_layer = 0xFFFFFFFF` (§14). They are not TEXT_CORE kinds.

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

Values `8..9`, `21..29`, `37..39`, `43..49`, `54..59`, and `81..95` are reserved (unused). A loader MUST
reject a block whose `source_kind` is not a defined value for its `module_kind` (or `OTHER` for a fused
block). The TEXT_CORE assignment is in §10/§14; the independent LM_HEAD_DRAFT assignment is in §14;
the MTP_DRAFT assignment is in §10.1/§14/§16; and the VISION_ENCODER assignment follows the §14 class
policy.

## 8. Quantization values (weight-only, symmetric, no zero point)

The format stores, **verbatim**, the quantizer's per-group output; it never recomputes scales or codes
(invariant 0.1). For each group of `group_size` consecutive K-elements of one row the quantizer
produces an fp16 scale `scale16` and `group_size` signed integer codes `q_i`, with dequant
`w_i = scale16 * q_i`. The **default** quantizer is per-group symmetric max-abs:

```text
qmax    = 2^(bits-1) - 1           # Q4=7, Q5=15, Q6=31, W8=127
scale   = max(abs(group)) / qmax   # fp32
scale16 = fp16(scale)              # stored AND used to quantize
q_i     = clamp(round_half_even(w_i / scale16), qmin, qmax)
```

`qmin = -(qmax+1)` for Q4/Q5/Q6; `qmin = -127` for W8G32. All-zero group: `scale = 0`,
codes `0`. A nonzero group whose scale underflows fp16 is bumped to the smallest positive fp16
subnormal.

When a **calibrated** pass runs, it MAY choose `scale16` per group by a different criterion (e.g.
error-minimizing search) while keeping the same per-group, symmetric, fp16-scale, no-zero-point
structure; the converter then sets header `flags.CALIBRATED`. Either way the stored `scale16`/`q_i` are
the quantizer's final output and the format reproduces them exactly. The format defines the *container*
for these values, not the algorithm that chose them.

**Value source per tensor.** Almost every quantized block's `w_i` are the model's canonical weights for
that projection (the same numbers regardless of storage). The one exception is the optional draft
`lm_head` (§14): its `w_i` are the **original bf16 `lm_head.weight` rows** of the shortlisted vocab
subset, quantized fresh at `Q4G64` (`qmax = 7`, `qmin = -8`) — i.e. re-derived from bf16, not shared
with the full `Q6` `lm_head` block. The format still stores that quantizer output verbatim (invariant
0.1 holds for the draft-head tensor's own values). The shortlist selection and id-map are described in
[2026-07-06-lm-head-draft-q4-decision.md](2026-07-06-lm-head-draft-q4-decision.md).

## 9. Payload encodings

### 9.1 Code packing (bit-plane)

A signed code `q_i` is stored as its `bits`-wide two's-complement pattern
`u_i = q_i & ((1 << bits) - 1)`. `u_i` is split into a **low nibble** and the **high bits**:

```text
low_i  = u_i & 0x0F                 # bits 0..3
high_i = u_i >> 4                   # Q5: 1 bit (0..1); Q6: 2 bits (0..3); Q4: none; W8G32: see below
```

Per group of `group_size` codes, the format emits:

- **Nibble run** (the low-4-bit base plane) — `group_size / 2` bytes. Byte `b` (`0 <= b < group_size/2`)
  holds `low_{2b} | (low_{2b+1} << 4)`. For **Q4** this is the whole code.
- **High run** (the high-bit plane) — present only for Q5/Q6:
  - **Q5**: 1 high bit per code -> `group_size / 8` bytes (8 for `group_size=64`). The high bits are
    packed LSB-first: high bit of code `c` is at bit `c & 7` of byte `c >> 3`.
  - **Q6**: 2 high bits per code -> `group_size / 4` bytes (16 for `group_size=64`). Packed LSB-first,
    two consecutive bits per code: bits `2c` and `2c+1` of the run are `high_c & 1` and
    `(high_c >> 1) & 1` (i.e. bits 4 and 5 of `u_c`). Reconstruct `high_c = (run >> (2c)) & 0x3`.
- **W8G32** has **no nibble split and no high run**: the base ("nibble") plane stores one signed
  `int8` per code, 32 bytes per group; `high_i` is unused. One FP16 scale covers each group of 32
  codes.

Per-group byte counts:

| qtype | bits | base ("nibble") bytes/group | high bytes/group | total bytes/group |
|---|---:|---:|---:|---:|
| Q4 | 4 | 32 | 0 | 32 |
| Q5 | 5 | 32 | 8 | 40 |
| Q6 | 6 | 32 | 16 | 48 |
| W8G32 | 8 | 32 (one `int8` per code) | 0 | 32 |

**Decode (reconstruction):**

```text
# Q4/Q5/Q6:
u_c = low_c | (high_c << 4)         # Q5: 5-bit; Q6: 6-bit; Q4: u_c = low_c
q_c = sign_extend(u_c, bits)        # Q4 from bit3, Q5 from bit4, Q6 from bit5
# W8G32:
q_c = (int8) base_byte_c            # already signed; no merge, no sign-extend step
# all qtypes:
w_c = scale16(group) * q_c
```

Worked Q5 example: quantizer code `q = -5`. `u = -5 & 0x1F = 27 = 0b1_1011`. `low = 0b1011 = 11`,
`high = 0b1 = 1`. Stored: nibble `11` in the nibble run, bit set in the high run. Decode:
`u = 11 | (1 << 4) = 27`; `sign_extend(27, 5) = 27 - 32 = -5`. Correct.

Worked Q4 example (draft head): quantizer code `q = -3`. `u = -3 & 0x0F = 13 = 0b1101`, no high bits.
Stored: nibble `13`. Decode: `u = 13`; `sign_extend(13, 4) = 13 - 16 = -3`. Correct.

### 9.2 `ROW_SPLIT` (every quantized block)

Logical `[N, K]`. `K` is padded to **`K_pad = align_up(K, 128)`**. This is the single mandatory
K-alignment rule for every quantized qtype; it is independent of `group_size`. It makes
`G = K_pad / group_size` is integral for every qtype and even for the group-64 qtypes. Code/high-plane
row runs preserve the vector-addressing alignment required by the format; scale runs are dense FP16
arrays and are not promised a 16-byte row stride. `N` is **not** padded. Let `nib` =
32 bytes/group for every qtype and `hi` = high bytes/group
(`0` Q4/W8G32, `8` Q5, `16` Q6). The payload is **three row-major planes** over all `N` rows
(Q4/W8G32: two — the high plane is empty), each starting 256-aligned:

```text
# nibble (base) plane: low-4-bit (or int8 for W8G32) codes, row-major
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
high_plane_bytes   = N * G * hi                          # 0 for Q4 / W8G32
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
  vector-width nibble loader serves Q4, Q5, and Q6 nibble planes (including the draft-head Q4 block).
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
| `lm_head_draft` | `[131072,5120]` | Q4 | 80 | 335,544,320 | 0 | 20,971,520 | `LM_HEAD_DRAFT[0,131072)` |
| `ATTN_IN` Q4 block | `[7168,5120]` | Q4 | 80 | 18,350,080 | 0 | 1,146,880 | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)` |
| `ATTN_IN` Q5 block | `[7168,5120]` | Q5 | 80 | 18,350,080 | 4,587,520 | 1,146,880 | `ATTN_GATE[0,6144)`, `ATTN_V[6144,7168)` |
| `GDN_IN` Q4 block | `[4096,5120]` | Q4 | 80 | 10,485,760 | 0 | 655,360 | `GDN_IN_PROJ_Q[0,2048)`, `GDN_IN_PROJ_K[2048,4096)` |
| `GDN_IN` Q5 block | `[6144,5120]` | Q5 | 80 | 15,728,640 | 3,932,160 | 983,040 | `GDN_IN_PROJ_V[0,6144)` |
| `MLP_GATEUP` block | `[34816,5120]` | Q4 | 80 | 89,128,960 | 0 | 5,570,560 | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` |
| `mtp.attn_in.w8` | `[14336,5120]` | W8G32 | 160 | 73,400,320 | 0 | 4,587,520 | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)`, `ATTN_GATE[7168,13312)`, `ATTN_V[13312,14336)` |
| `mtp.mlp.gateup.w8` | `[34816,5120]` | W8G32 | 160 | 178,257,920 | 0 | 11,141,120 | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` |

For `lm_head_draft`: `K_pad = align_up(5120,128) = 5120`, `G = 5120/64 = 80`, `nib = 32`, `hi = 0`, so
`nibble = 131072*80*32 = 335,544,320`, `high = 0`, `scale = 131072*80*2 = 20,971,520`,
`payload_bytes = align_up(335,544,320,256) + 20,971,520 = 356,515,840` (the nibble plane is already a
multiple of 256). Total code bytes per block (`nibble + high`) equal `N*G*(nib+hi)` — `40·N·G` Q5,
`48·N·G` Q6 and `32·N·G` Q4/W8G32 — i.e. `bytes_per_token` for a full stream is unchanged
by the plane split.

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

`high_ptr == nullptr` signals a single-plane block (Q4/W8G32, including the draft head); a
consumer of such a block uses only the base plane. These pointer/offset rules — not the kernel
internals — are the ABI a conformant GEMV/GEMM relies on; they are fully determined by the
`TensorEntry` plane-byte fields and §9.2.

### 9.4 `CONTIGUOUS` (BF16_CTRL / FP32_CTRL / I32_CTRL)

Raw row-major elements in the stated dtype, little-endian. `group_size=0`, `scale_dtype=none`,
`nibble_plane_bytes = payload_bytes`, `high_plane_bytes = 0`, `scale_plane_bytes = 0`, one segment
spanning the whole tensor.

- `BF16_CTRL` / `FP32_CTRL`: 2- / 4-byte IEEE-754 elements. Used by norms, GDN `A_log`/`dt_bias`,
  conv1d, and the `[48,5120]` GDN `in_a`/`in_b` dense-control gates.
- `I32_CTRL`: 4-byte signed two's-complement `int32` elements. Used by the draft-head id-map
  `lm_head_draft.idmap` (`[131072]`, `payload_bytes = nibble_plane_bytes = 131072*4 = 524,288`, one
  segment `LM_HEAD_DRAFT_IDMAP[0,131072)`). Entry `i` is the real vocab id of shortlist row `i`; the
  runtime remaps a draft argmax over the `N` shortlisted rows back to a vocab id through this map.

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
  `MLP_DOWN`, `LM_HEAD`, `EMBED`, and all TEXT_CORE control tensors. The optional draft-head pair is
  standalone in its own `LM_HEAD_DRAFT` module (§14), not part of TEXT_CORE.

### 10.1 Fused projection groups (MTP_DRAFT)

MTP_DRAFT reuses the existing module-agnostic fusion ids. Its member blocks are distinguished from
TEXT_CORE by `TensorEntry.module_kind = MTP_DRAFT`; no new fusion id is introduced.

| group | source_layer | member blocks (`qtype`, `[N,K]`) | segments (row-ranges) | `total_n` |
|---|---:|---|---|---:|
| `ATTN_IN` | 0 | W8G32 `[14336,5120]` named `mtp.layers.0.attn_in.w8` | `ATTN_Q[0,6144)`, `ATTN_K[6144,7168)`, `ATTN_GATE[7168,13312)`, `ATTN_V[13312,14336)` | 14336 |
| `MLP_GATEUP` | 0 | W8G32 `[34816,5120]` named `mtp.layers.0.mlp.gateup.w8` | `MLP_GATE[0,17408)`, `MLP_UP[17408,34816)` | 34816 |

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
  logical projection by segment (`row_begin..row_begin+row_count` -> `source_kind`). The consumer reads
  each row's base run (vector-width), its high run (Q5/Q6), and its scales, and reconstructs each code
  as in §9.1 before the multiply-accumulate.
- The draft `lm_head` is consumed exactly as any other standalone `ROW_SPLIT` Q4 block: a GEMV over its
  `N` shortlisted rows produces `N` logits; the argmax row index is mapped to a vocab id through the
  `LM_HEAD_DRAFT_IDMAP` block (§9.4). Verification uses the full `LM_HEAD` block, not the draft head.
- A fusion group is executed as one GEMV per member block (i.e. per `qtype`), each over its block's
  rows, scattering each segment's rows to its logical projection output. The shared input is read once
  per member block (not once per projection) and is reused across the group's blocks via L2; the format
  enables this shared-input dispatch but does not mandate a single physical input load.
- Prefill GEMM consumes the same blocks by staging `[row-range x k-range]` segments into SMEM
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
  When `high_plane_bytes == 0` (Q4/W8G32) the scale plane immediately follows the aligned
  nibble plane. All inter-plane pad bytes are zero (§0).
- **CONTIGUOUS.** A `CONTIGUOUS` block (BF16/FP32/I32) has a single payload run of `payload_bytes`; no
  K-padding, no inter-plane pad.
- **Block / member alignment.** Each block payload is 256-aligned. Fusion-group member blocks are
  emitted consecutively, each 256-aligned; `FusionGroupRecord.payload_bytes` covers the inter-member
  (zero) padding.

## 13. Structural validation (loader / verifier conformance)

A conformant loader MUST reject a file that fails any of the **structural** checks 1–8 (a failure there
can mis-read weights or change token identity). Checks marked *(verifier)* — 9 `name_hash`, 10 full
weight zero-fill, 11 weight crc/dequant — are **integrity** checks for the offline auditor; a loader MAY
limit them to selected payloads at load time. Header/catalog/tokenizer reserved bytes and tokenizer
CRC are always runtime-mandatory. All checks are computable from this document alone.

1. **Header.** `magic == "Q5090MIXEDV4\0\0\0\0"`; `version == 4`; `endian == 0x01020304`;
   `header_size == 4096`; `format_minor == 2`; `module_count in [1,4]`; `layer_count == 64`;
   `tokenizer_record_count == 3`; `tokenizer_record_size == 64`; `tokenizer_flags == 0`. Header
   reserved bytes and flag bits 5..31 are zero. Presence bits agree with the module table (§2).
   `tensor_count <= 2048`, `segment_count <= 4096`, `fusion_group_count <= 512`, and
   `string_table_bytes <= 64 MiB`.
   `format_minor == 0` and all unknown minors are rejected after the 4096-byte header read.
2. **Index adjacency / sizes.** The module/tensor/segment/fusion/string offsets satisfy the adjacency
   chain of §1, and `module_index_bytes == module_count*64`,
   `tensor_index_bytes == tensor_count*128`, `segment_index_bytes == segment_count*32`,
   `fusion_group_index_bytes == fusion_group_count*64`.
   `tokenizer_index_offset == align_up(string_table_offset + string_table_bytes,64)`,
   `tokenizer_index_bytes == 192`, and
   `tokenizer_data_offset == align_up(tokenizer_index_offset + 192,64)`. Every offset/size calculation
   uses checked u64 arithmetic and lies within `file_size`.
3. **Tokenizer.** Records have the exact §2.1 order/kind/encoding, non-empty bounded ranges, correct
   64-byte placement, no overlap, and `tokenizer_data_bytes` ends at the final asset end.
   `payload_offset == align_up(tokenizer_data_offset + tokenizer_data_bytes,4096)`. The runtime checks
   tokenizer CRC32, UTF-8, and every tokenizer-region padding byte. The offline verifier also checks
   SHA-256 and tokenizer semantics.
4. **Modules.** `module_kind`s are distinct and follow the canonical
   TEXT->LM_HEAD_DRAFT->MTP->VISION order with absent optional modules omitted; their
   `[tensor_index_begin, +tensor_index_count)` ranges partition `[0, tensor_count)` contiguously. Each
   module's offset-40/44 u32 and remaining reserved bytes are zero, and its
   `payload_offset/payload_bytes` span covers exactly its blocks' payloads. TEXT_CORE contains no
   draft source kinds.
5. **Payload region.** Each block `payload_offset` is 256-aligned and lies in
   `[file payload_offset, file_size)`; blocks appear in **tensor-index order** with no overlap; block
   `payload_offset + payload_bytes <= next block payload_offset` (or `file_size` for the last); the
   header `payload_bytes` and `file_size` are consistent with the last block's end.
6. **Per-block fields.** `group_size`/`scale_dtype` match `qtype` (Q4/Q5/Q6: 64/FP16;
   W8G32: 32/FP16; BF16_CTRL/FP32_CTRL/I32_CTRL: 0/none). Then, **by layout**:
   - **ROW_SPLIT**: `padded_shape[0] == shape[0] == N`, `padded_shape[1] == align_up(shape[1],128)`,
     `shape[2]==shape[3]==1`; `nibble_plane_bytes`/`high_plane_bytes`/`scale_plane_bytes` equal the §9.2
     formulas; `payload_bytes == scale_rel + scale_plane_bytes`.
   - **CONTIGUOUS** (BF16_CTRL/FP32_CTRL/I32_CTRL): `padded_shape == shape` (all four dims, no
     K-padding); `high_plane_bytes == 0`, `scale_plane_bytes == 0`, `nibble_plane_bytes == payload_bytes
     == elem_bytes * product(shape)` (`elem_bytes` = 2/4/4 for BF16/FP32/I32).
7. **Segments.** `[segment_begin, segment_begin+segment_count) subset [0, segment_count_total)`; a
   block's segments partition `[0, N)` exactly with strictly increasing `row_begin` and no gap/overlap;
   if `segment_count == 1`, block `source_kind/source_layer == segment[0]`'s; if `segment_count > 1`,
   block `source_kind == OTHER (0)`. Each `source_kind` is a defined value for its module (§7). The
   `LM_HEAD_DRAFT` block has one segment `LM_HEAD_DRAFT[0,N)`; the `LM_HEAD_DRAFT_IDMAP` block has one
   segment `LM_HEAD_DRAFT_IDMAP[0,N)`; both use `source_layer == 0xFFFFFFFF`.
8. **Fusion groups.** Members are `block_count` consecutive blocks at `first_block_tensor_index`,
   consecutive in payload; each member's `fusion_group_id == group_id`, equal `source_layer`, equal
   `K`, `fusion_index == position`; `total_n == sum member N`; `shared_k == K`.
9. **String table.** Every `name_len` is in `[1,4096]`; the name is valid UTF-8 with no embedded NUL;
   `name_offset + name_len < string_table_bytes`; the byte at `name_offset + name_len` is `NUL`; and
   `name_hash == FNV-1a-64(name)` (§15). The runtime enforces the length/range/encoding/hash checks
   before materializing owned name strings, preventing bounded metadata from amplifying into
   unbounded host allocations.
10. **(verifier) Weight zero-fill.** All string-table/weight inter-plane/inter-block padding bytes are
   zero (§0). Header/catalog/tokenizer reserved and padding bytes are structural and already covered
   by checks 1–4.
11. **(verifier)** `crc32` matches the block payload (§15); dequantizing each block reproduces the
    quantizer values bit-for-bit (invariant 0.1).

**Draft-head coupling (structural).** If `LM_HEAD_DRAFT_PRESENT` is set, exactly one
`LM_HEAD_DRAFT` module exists and contains exactly two blocks in order: one Q4G64 `ROW_SPLIT`
`LM_HEAD_DRAFT`, then one I32_CTRL `CONTIGUOUS` `LM_HEAD_DRAFT_IDMAP`. Both use
`source_layer=0xFFFFFFFF`, each has one full-range segment, and their `N` values match. Every id-map
entry is unique and `< FileHeader.vocab_size`. If the flag is clear, the module and both source kinds are absent.
A loader MUST reject any other combination.

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
| `lm_head_draft` (optional module) | Q4 | `ROW_SPLIT` | standalone in `LM_HEAD_DRAFT` (§18) |
| `lm_head_draft.idmap` (optional module) | I32_CTRL | `CONTIGUOUS` | standalone in `LM_HEAD_DRAFT` |
| norms, `A_log`, `dt_bias`, conv1d | BF16/FP32 | `CONTIGUOUS` | standalone |
| MTP `fc`, `o_proj`, `down_proj` | W8G32 | `ROW_SPLIT` (group 32) | standalone |
| MTP `attn q/k/gate/v` | W8G32 | `ROW_SPLIT` (group 32) | `ATTN_IN` (one block, four segments) |
| MTP MLP `gate`, `up` | W8G32 | `ROW_SPLIT` (group 32) | `MLP_GATEUP` (one block, two segments) |
| vision patch embed | Q6 | `ROW_SPLIT` | standalone; canonical Conv3d flatten below |
| vision block `qkv` / `fc1` | Q4 | `ROW_SPLIT` | standalone |
| vision block `proj` / `fc2` | Q5 | `ROW_SPLIT` | standalone |
| vision merger `fc1` / `fc2` | W8G32 | `ROW_SPLIT` (group 32) | standalone |
| vision position table, biases, norms | BF16_CTRL | `CONTIGUOUS` | standalone |

Every quantized tensor in every module uses `ROW_SPLIT`; every dense control tensor uses `CONTIGUOUS`.
There is no tiled layout anywhere. The class table above fixes layout/grouping. The per-layer TEXT_CORE
and VISION_ENCODER emission follows mechanically from this policy applied across the 64 decoder layers
(and the vision blocks); the TEXT_CORE globals are only `EMBED`, `LM_HEAD`, and `FINAL_NORM`. The
LM_HEAD_DRAFT and MTP_DRAFT assignments are the canonical block orders that follow.

The patch source has shape `[1152,3,2,16,16]` in contiguous `[out,C,T,H,W]` order. The converter
flattens it directly to `[1152,1536]`; therefore K order is `C,T,H,W` with W varying fastest. A
consumer MUST construct each patch vector in that same order. The format defines no Conv3d-specific
layout and stores no second copy.

**Optional `LM_HEAD_DRAFT` module.** When `LM_HEAD_DRAFT_PRESENT` is set, the independent module emits
exactly these two standalone blocks, the id-map immediately after the weights:

| block name | qtype/layout | shape | source kind / segments |
|---|---|---:|---|
| `lm_head_draft` | Q4G64 row-split | `[131072,5120]` | `LM_HEAD_DRAFT`, `source_layer 0xFFFFFFFF`; one segment `LM_HEAD_DRAFT[0,131072)` |
| `lm_head_draft.idmap` | I32_CTRL contiguous | `[131072]` | `LM_HEAD_DRAFT_IDMAP`, `source_layer 0xFFFFFFFF`; one segment `LM_HEAD_DRAFT_IDMAP[0,131072)` |

The `lm_head_draft` codes/scales are the `Q4G64` quantization of the original bf16 `lm_head.weight` rows
selected by a frequency-ranked shortlist (plus force-included special ids); the id-map entries are those
rows' vocab ids (§8, §9.4). The shortlist size (`N = 131072`), selection method, and measured
justification are in [2026-07-06-lm-head-draft-q4-decision.md](2026-07-06-lm-head-draft-q4-decision.md).

**MTP_DRAFT canonical block order:**

| block name | qtype/layout | shape | source kind / segments |
|---|---|---:|---|
| `mtp.fc.weight` | W8G32 row-split | `[5120,10240]` | `MTP_FC`, `NO_LAYER` |
| `mtp.pre_fc_norm_embedding.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_PRE_FC_NORM_EMB`, `NO_LAYER` |
| `mtp.pre_fc_norm_hidden.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_PRE_FC_NORM_HID`, `NO_LAYER` |
| `mtp.layers.0.input_layernorm.weight` | BF16_CTRL contiguous | `[5120]` | `INPUT_LAYERNORM`, layer 0 |
| `mtp.layers.0.attn_in.w8` | W8G32 row-split | `[14336,5120]` | block `OTHER`; segments `ATTN_Q/K/GATE/V`, layer 0 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16_CTRL contiguous | `[256]` | `ATTN_Q_NORM`, layer 0 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16_CTRL contiguous | `[256]` | `ATTN_K_NORM`, layer 0 |
| `mtp.layers.0.self_attn.o_proj.weight` | W8G32 row-split | `[5120,6144]` | `ATTN_O`, layer 0 |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16_CTRL contiguous | `[5120]` | `POST_ATTN_LAYERNORM`, layer 0 |
| `mtp.layers.0.mlp.gateup.w8` | W8G32 row-split | `[34816,5120]` | block `OTHER`; segments `MLP_GATE/MLP_UP`, layer 0 |
| `mtp.layers.0.mlp.down_proj.weight` | W8G32 row-split | `[5120,17408]` | `MLP_DOWN`, layer 0 |
| `mtp.norm.weight` | BF16_CTRL contiguous | `[5120]` | `MTP_NORM`, `NO_LAYER` |

A converter or verifier implements these assignments to produce/validate the concrete
block/segment/fusion set. A full artifact with all four modules contains 1166 blocks, 1314 segments,
and 130 fusion groups: TEXT_CORE contributes 819/963/128, LM_HEAD_DRAFT contributes 2/2/0,
MTP_DRAFT contributes 12/16/2, and VISION_ENCODER contributes 333/333/0. With LM_HEAD_DRAFT absent,
the totals are 1164/1312/130; `fusion_group_count` is unchanged.

`embed_tokens` is gather-driven (one row per token), not GEMV-streamed, so the row/plane split gives it
no throughput benefit and costs a couple of extra small transactions per lookup (a row's nibble run,
high run, and scales are in three planes rather than inline). It uses `ROW_SPLIT` anyway as a
deliberate uniformity choice: the cost is negligible (a few extra bytes of latency on a ~4 KB lookup,
dwarfed by per-token weight streaming), it keeps a single quantized layout, and it lets `embed` and
`lm_head` (identical `[248320,5120] Q6`) share one layout. The draft `lm_head`, by contrast, **is**
GEMV-streamed at each draft site, so its Q4 shortlist directly cuts draft-step bandwidth (§18).

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
results against the quantizer reference (invariant 0.1), independent of byte order.

## 16. Canonical names for sliced/reshaped source tensors

Source-model tensors that map to multiple projections, or multiple source tensors that map into one
fused block, are reconciled by the segment table; each `SegmentRecord` carries the projection's
canonical `name`/`name_hash` (§5). Per-tensor names follow from the assignment policy (§10, §14);
representative cases:

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

lm_head.weight [248320,5120]  -> LM_HEAD [248320,5120] Q6 (standalone block)
  # optional draft head (shortlist of the same source rows, re-quantized at Q4):
  -> LM_HEAD_DRAFT       [131072,5120] Q4      (standalone block; rows = frequency shortlist)
  -> LM_HEAD_DRAFT_IDMAP [131072]      I32_CTRL (standalone block; entry i = vocab id of draft row i)

mtp.layers.0.self_attn.q_proj.weight [12288,5120]   # interleaved [query(256)|gate(256)] x24
  -> ATTN_Q   [6144,5120] W8G32 (MTP ATTN_IN block segment)
  -> ATTN_GATE[6144,5120] W8G32 (MTP ATTN_IN block segment)
mtp.layers.0.self_attn.k_proj.weight -> ATTN_K [1024,5120] W8G32 (MTP ATTN_IN block segment)
mtp.layers.0.self_attn.v_proj.weight -> ATTN_V [1024,5120] W8G32 (MTP ATTN_IN block segment)

mtp.layers.0.mlp.gate_proj.weight -> MLP_GATE [17408,5120] W8G32 (MTP MLP_GATEUP block segment)
mtp.layers.0.mlp.up_proj.weight   -> MLP_UP   [17408,5120] W8G32 (MTP MLP_GATEUP block segment)
```

Each block's `name`/`name_hash` is the converter-assigned canonical name (for a standalone block, the
projection name — e.g. `lm_head_draft` / `lm_head_draft.idmap`; for a fused block, the group name, e.g.
`layers.{L}.attn_in.q4` for TEXT_CORE or `mtp.layers.0.attn_in.w8` for MTP_DRAFT); per-projection names
are recovered from segment `name_*`.

## 17. Manifest

The sidecar `manifest.json` (file name `<weights>.manifest.json`, next to the `.qus`) is an
informational mirror of the file the converter emits; the binary header is authoritative. Required
fields:

```json
{
  "format": "q5090_w4g64_mixed_v4_2",
  "format_version": 4,
  "format_minor": 2,
  "binary_spec": "docs/q5090_packed_file_format_v4.md",
  "value_source": "per-group symmetric weight-only quant (§8); MTP dense/fused linears W8G32; optional draft lm_head = Q4G64 of original bf16 lm_head shortlist rows",
  "weights_file": "qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus",
  "file_bytes": 0,
  "sha256_safetensors_index": "<64 hex chars>",
  "calibrated": false,
  "lm_head_draft_present": true,
  "alignment": { "header": 4096, "tokenizer": 64, "payload": 4096, "block": 256, "k_pad": 128, "group_sizes": [32, 64] },
  "layouts": ["ROW_SPLIT", "CONTIGUOUS"],
  "code_planes": ["nibble", "high", "scale"],
  "qtypes": ["Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S", "BF16_CTRL", "FP32_CTRL", "I32_CTRL"],
  "modules": ["TEXT_CORE", "LM_HEAD_DRAFT", "MTP_DRAFT", "VISION_ENCODER"],
  "absent_modules": [],
  "tensor_count": 1166,
  "segment_count": 1314,
  "fusion_group_count": 130,
  "tokenizer": {
    "record_count": 3,
    "assets": [
      {"kind": "TOKENIZER_JSON", "bytes": 0, "crc32": "00000000", "sha256": "<64 hex chars>"},
      {"kind": "MERGES_TXT", "bytes": 0, "crc32": "00000000", "sha256": "<64 hex chars>"},
      {"kind": "GENERATION_CONFIG_JSON", "bytes": 0, "crc32": "00000000", "sha256": "<64 hex chars>"}
    ]
  },
  "lm_head_draft": {
    "present": true,
    "module": "LM_HEAD_DRAFT",
    "n": 131072,
    "k": 5120,
    "weights_qtype": "Q4G64_F16S",
    "weights_source_kind": "LM_HEAD_DRAFT",
    "weights_payload_bytes": 356515840,
    "idmap_qtype": "I32_CTRL",
    "idmap_source_kind": "LM_HEAD_DRAFT_IDMAP",
    "idmap_payload_bytes": 524288,
    "selection": "docs/2026-07-06-lm-head-draft-q4-decision.md"
  },
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
MUST equal the header values (a block is one `TensorEntry`, so `tensor_count` is the block count);
`modules` lists the modules present in this file and `absent_modules` the supported-but-omitted ones;
`lm_head_draft_present` MUST equal header `flags.LM_HEAD_DRAFT_PRESENT`; when the module is absent,
drop the `lm_head_draft` object (or set `present: false`), list `LM_HEAD_DRAFT` in `absent_modules`, and
lower `tensor_count`/`segment_count` by 2 (`1164`/`1312`). Tokenizer asset bytes/hashes MUST equal the
three TokenizerRecords. `effective_text_bpw` is the quantizer value. The manifest is not consumed by
the runtime loader but MUST be exactly consistent with the authoritative binary header/catalog.

## 18. Non-goals and scope

- **No general bytes/token reduction for the shared weights.** For the Q4/Q5/Q6/W8G32 tensors, the
  format stores the same quantizer codes/scales; the gain is achievable `BW_eff` and reduced
  launch/occupancy/unpack waste, not less traffic.
- **The draft head is the one bandwidth-reducing tensor.** It is a *smaller, distinct* head (a
  frequency shortlist of vocab rows at Q4), not a relayout of the full head. It reduces draft-step
  traffic; it is optional and orthogonal to module presence.
- **Verify always uses the full `lm_head`.** The draft head only changes which tokens are *proposed*
  during MTP drafting; every *emitted* token is decided by the full `LM_HEAD`. A poorer draft lowers
  acceptance length only; it can never change correctness.
- **Draft-head values re-derived from bf16 (only place).** The `lm_head_draft` codes/scales are `Q4G64`
  of the original bf16 `lm_head.weight` rows, not shared with the full `Q6` `lm_head`. This is the sole
  tensor whose values are freshly derived rather than shared; the format still stores that quantizer
  output verbatim (invariant 0.1 for that tensor).
- **No low-bit numeric-format change.** No FP4/FP6/MXFP/NVFP; values equal the active quantizer for
  each qtype. `W8G32_F16S` is still signed int8 with FP16 per-group scales, but uses a 32-element
  group. `I32_CTRL` is raw int32 control data, not a quantized numeric format.
- **No second resident copy and no dequant-to-BF16/FP32** of any quantized weight.
- **Per-group fp16 scales.** One fp16 scale per `(row, group)` of 64 for Q4/Q5/Q6 or 32 for W8G32.
  The format does not use hierarchical/sub-block quantized scales.
- **Float-accumulation consumption assumed.** The byte layout targets a dequantize-to-float GEMV/GEMM;
  it neither requires nor encodes an integer-dot (quantized-activation) execution path.
- **Decode-first prefill, explicitly.** The global byte order is GEMV-optimal; prefill GEMM consumes it
  via SMEM-staged per-row segment loads. Because prefill is compute-bound and lower priority, this is a
  deliberate choice, not a regression; the format does not store a GEMM-optimal duplicate.
- **No legacy layout, no fallback, no migration path.** Only `ROW_SPLIT` + `CONTIGUOUS` exist.
- **MTP_DRAFT / VISION_ENCODER / LM_HEAD_DRAFT are first-class but selectively present.** The format,
  `module_kind`/`source_kind` enums (§7), and converter support all three optional modules. Runtime
  feature requests decide residency; ModuleRecord contains no load policy. Module metadata remains
  available even when its payload is not resident.

## 19. v4.2 feature set

The format's distinctive capabilities, stated as its own feature list:

- **Bit-plane code packing.** Quantized codes are stored as a dense low-4-bit nibble plane plus a
  separate high-bit plane (Q5: 1 bit, Q6: 2 bits; Q4/W8: none). Decode is `sign_extend(low | high<<4)`
  — register-only shift/mask, no cross-byte funnel shift, no cross-lane communication. One nibble
  loader serves Q4, Q5, Q6 (§9.1).
- **Three-plane `ROW_SPLIT`.** Every quantized block stores nibble / high (empty for Q4/W8) / scale as
  three row-major, 256-aligned planes; the single K-padding rule is `K_pad = align_up(K,128)` (§9.2).
- **Self-describing plane sizes.** `nibble_plane_bytes`/`high_plane_bytes`/`scale_plane_bytes` and the
  relative-offset/`payload_bytes` definition are normative in every `TensorEntry` (§4, §9.2, §9.3).
- **One W8 format.** MTP_DRAFT dense/fused linears and both vision merger FC weights use
  `W8G32_F16S`; no W8G128 qtype or loader branch exists (§10.1, §14).
- **Vision precision policy.** The flattened patch embedding is Q6G64, block qkv/fc1 are Q4G64,
  block proj/fc2 are Q5G64, and controls remain BF16 (§14).
- **Embedded CPU-only tokenizer.** Three mandatory raw UTF-8 assets have fixed TokenizerRecords,
  bounds, placement, CRC32, and SHA-256 and are consumed without an external runtime path (§2.1).
- **Optional shortlisted draft `lm_head`.** A standalone `[131072,5120]` `Q4G64` `ROW_SPLIT` block
  in the independent `LM_HEAD_DRAFT` module, Q4-quantized from the original bf16 `lm_head` shortlist
  rows, cutting per-draft-step bandwidth (§14, §18).
- **`int32` draft id-map.** A paired `[131072]` `I32_CTRL` `CONTIGUOUS` block
  (`source_kind = LM_HEAD_DRAFT_IDMAP`) remaps a draft argmax over shortlist rows to a real vocab id
  (§9.4).
- **`LM_HEAD_DRAFT_PRESENT` header flag** (bit4) with structural coupling: set iff the independent
  two-block module is present, and the id-map `N` must equal the weights `N` (§2, §13).
- **Compact qtype enum.** Tags 0..6 are Q4G64, Q5G64, Q6G64, W8G32, BF16, FP32, and I32; there are
  no retired tag holes (§7).
- **Container identity.** `magic = "Q5090MIXEDV4\0\0\0\0"`, `version = 4`,
  `format = q5090_w4g64_mixed_v4_2`, `format_minor = 2`, `module_version = 4` (§2, §3).
- **Strict invariants.** Value preservation, single resident copy, on-chip dequant only, one layout
  family for decode+prefill, fully self-describing metadata — verifiable per §0 and §13.
