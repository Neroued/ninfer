# NInfer Persistent Storage Layouts

> Status: accepted on 2026-07-14, implementation in progress.
>
> Authority: this document registers the first two persistent tensor layouts and the first required-
> resource encoding used by `.ninfer` artifacts. It defines their supported numeric formats, file
> alignment, byte order, physical padding, exact encoded-size rules, and logical decode. Numeric
> values and grouped-quantization semantics come from
> [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md); container framing and object ranges come
> from [`ninfer-container-format.md`](ninfer-container-format.md). Model object inventories, source
> mappings, conversion recipes, logical model views, and execution policy are defined elsewhere.

## 1. Registered identities

The initial storage registry contains exactly these identities:

| Identity | Kind | Compatible numeric formats | Logical shape | Object alignment |
|---|---|---|---|---:|
| `contiguous-le-v1` | tensor layout | `BF16`, `FP32`, `I32` | rank `0..16` | 256 bytes |
| `row-split-k128-v1` | tensor layout | `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, `W8G32_F16S` | rank 2 `[N,K]` | 256 bytes |
| `raw-bytes-v1` | resource encoding | not applicable | nonempty byte string | 1 byte |

These are closed identities, not templates. A format/layout combination not present in the table is
unsupported. In particular, a direct format cannot use `row-split-k128-v1`, and a grouped format
cannot use `contiguous-le-v1`.

Object alignment applies to the object's payload-relative `offset` in the `.ninfer` JSON. Internal
plane offsets and padding belong to the selected layout. Inter-object padding belongs to the
container and is not included in an object's `bytes`.

The helper used below is:

```text
align_up(x, a) = ceil_div(x, a) * a
```

All formula inputs and intermediate results are nonnegative integers. The container implementation
must reject a shape or calculation that cannot be represented by its file-offset and size types.

## 2. `contiguous-le-v1`

### 2.1 Logical traversal

`contiguous-le-v1` stores direct logical words in C order: the last logical dimension varies
fastest. For shape `[D0, D1, ..., D(r-1)]`, coordinate `[i0, i1, ..., i(r-1)]` has linear index:

```text
index = (((i0 * D1 + i1) * D2 + i2) ... ) * D(r-1) + i(r-1)
```

A rank-zero shape `[]` contains one scalar word. For every other legal shape:

```text
elements = product(shape)
```

The layout performs no reshape, transpose, type conversion, or numerical canonicalization. Those
operations, when required, occur before layout encoding under the model-specific conversion recipe.

### 2.2 Word bytes

Words are serialized least-significant byte first:

| Format | Bytes per element | Stored word |
|---|---:|---|
| `BF16` | 2 | the exact 16-bit bfloat16 logical word, little-endian |
| `FP32` | 4 | the exact 32-bit IEEE-754 binary32 logical word, little-endian |
| `I32` | 4 | the exact 32-bit two's-complement logical word, little-endian |

Signed zero, subnormal, infinity, NaN payload, and integer-word behavior are determined by the
numeric-format contract. The layout only preserves the word bits.

### 2.3 Encoded size

There is no internal prefix, stride table, per-row padding, or trailing padding:

```text
payload_bytes = elements * bytes_per_element(format)
```

The tensor object's JSON `bytes` must equal `payload_bytes` exactly.

## 3. `row-split-k128-v1`

### 3.1 Logical and physical geometry

The logical tensor is a positive rank-two matrix `[N,K]`. The format supplies code width `b` and
group size `G`:

| Format | `b` | `G` | Base bytes per group `B` | High bytes per group `H` |
|---|---:|---:|---:|---:|
| `Q4G64_F16S` | 4 | 64 | 32 | 0 |
| `Q5G64_F16S` | 5 | 64 | 32 | 8 |
| `Q6G64_F16S` | 6 | 64 | 32 | 16 |
| `W8G32_F16S` | 8 | 32 | 32 | 0 |

The layout extends the last axis to a multiple of 128:

```text
K_pad              = align_up(K, 128)
groups_per_row     = K_pad / G
logical_groups     = ceil_div(K, G)
physical_group_cnt = N * groups_per_row
```

`K_pad` is physical geometry and is not added to the JSON `shape`. Because both registered group
sizes divide 128, `groups_per_row` is integral.

For the final partially logical group, lanes whose column is at least `K` have signed code zero. Its
scale remains the scale of the logical group defined by the numeric-format contract. Any complete
physical group after `logical_groups` has scale word `0x0000` and all codes zero. Physical padding
therefore cannot change a decoded logical value.

### 3.2 Payload planes

The payload contains three conceptual planes in this order:

```text
base-code plane
zero padding to a 256-byte boundary
optional high-bit plane
zero padding to a 256-byte boundary
binary16 scale plane
```

Q4 and W8 have no high-bit bytes. They still place the scale plane at the first 256-byte boundary
after the base-code plane. There is no padding after the scale plane inside the object.

Within every plane, traversal order is:

```text
row 0 group 0, row 0 group 1, ..., row 1 group 0, ...
```

Bytes belonging to one group are adjacent.

### 3.3 Base-code plane

For Q4, Q5, and Q6, let `q[i]` be lane `i`'s signed code and let:

```text
u[i] = q[i] modulo 2^b
```

be its unsigned `b`-bit two's-complement word. Each consecutive lane pair occupies one base byte:

```text
base[j] = (u[2*j] & 0x0f) | ((u[2*j + 1] & 0x0f) << 4)
```

Thus the even lane is in the low nibble and the odd lane is in the high nibble. Every G64 group
occupies 32 base bytes.

For W8, each lane occupies one byte containing its exact 8-bit two's-complement word. Lane `i`
occupies byte `i`, so every G32 group occupies 32 base bytes. The numeric-format restriction that
excludes code `-128` remains in force.

The complete base plane is the concatenation of these per-group byte sequences in plane traversal
order.

### 3.4 High-bit plane

Only Q5 and Q6 have a high-bit plane. For each lane:

```text
high[i] = (u[i] >> 4) & ((1 << (b - 4)) - 1)
```

The high-bit stream is lane-major. Within one lane, bit 4 is emitted first, followed by bit 5 for
Q6. Stream bit `t` is stored in bit `(t mod 8)` of byte `floor(t / 8)`; bit zero is therefore the
first bit of every byte.

Equivalently:

- Q5 byte 0 contains the high bit of lanes 0 through 7 in byte bits 0 through 7;
- Q6 byte 0 contains lane 0 bits 4 and 5 in byte bits 0 and 1, lane 1 bits 4 and 5 in byte bits 2
  and 3, and so on.

One Q5 G64 group occupies 8 high bytes. One Q6 G64 group occupies 16 high bytes.

### 3.5 Scale plane

Every physical group owns one 16-bit scale word. Scale words follow the same row/group traversal as
the code planes and are stored little-endian:

```text
scale_index(row, group) = row * groups_per_row + group
```

For a logical group, the word is exactly the binary16 multiplier defined by its numeric format. The
padding rule in Section 3.1 defines the scale words for wholly physical groups.

### 3.6 Plane offsets and encoded size

Let:

```text
base_bytes  = N * groups_per_row * B
high_bytes  = N * groups_per_row * H
scale_bytes = N * groups_per_row * 2

base_offset  = 0
high_offset  = align_up(base_bytes, 256)
scale_offset = high_offset + align_up(high_bytes, 256)

payload_bytes = scale_offset + scale_bytes
```

The byte ranges are:

```text
base  = [base_offset,  base_offset  + base_bytes)
high  = [high_offset,  high_offset  + high_bytes)
scale = [scale_offset, scale_offset + scale_bytes)
```

When `H = 0`, `high_bytes = 0` and `scale_offset = high_offset`. Bytes between the end of the base
plane and `high_offset`, and between the end of a nonempty high plane and `scale_offset`, are zero.

The tensor object's JSON `bytes` must equal `payload_bytes`; it does not include the gap needed to
align the following object.

For example, a Q5 tensor with shape `[2,130]` has `K_pad=256`, four groups per row,
`base_bytes=256`, `high_bytes=64`, `scale_bytes=16`, `high_offset=256`, `scale_offset=512`, and
`payload_bytes=528`.

### 3.7 Row views and row slices

Row addressability is an intrinsic property of this layout. Define:

```text
base_row_bytes  = groups_per_row * B
high_row_bytes  = groups_per_row * H
scale_row_bytes = groups_per_row * 2
```

A non-owning logical view of consecutive rows `[row_begin, row_begin + row_count)` uses:

```text
base_view  = base_offset  + row_begin * base_row_bytes
high_view  = high_offset  + row_begin * high_row_bytes   # absent when H = 0
scale_view = scale_offset + row_begin * scale_row_bytes
```

and spans `row_count * base_row_bytes`, `row_count * high_row_bytes`, and
`row_count * scale_row_bytes` in the respective planes. Its logical shape is `[row_count,K]` and it
retains the parent object's `K_pad` and `groups_per_row`. A row view is therefore three plane spans,
not one assumed-contiguous payload range.

If those rows are materialized as a standalone payload, their three row spans are concatenated in
the same plane order, with the plane offsets and zero padding recomputed from Section 3.6 using
`N=row_count`. This produces another valid `row-split-k128-v1` tensor without decoding or repacking
individual codes.

## 4. `raw-bytes-v1`

`raw-bytes-v1` is a required-resource encoding, not a tensor layout. Its enclosing object payload is
the resource byte string itself:

```text
payload_bytes = resource_length
alignment     = 1
```

It has no embedded length, header, terminator, filename, character encoding, compression, or
trailing padding. The resource object's JSON `bytes` is its exact nonzero length, and a reader
returns the complete span unchanged. A model contract assigns a resource name and interprets those
bytes; the common encoding does not infer that meaning from the name.

## 5. Decode boundary

Layout decoding yields only persistent logical words:

- `contiguous-le-v1` yields the direct BF16, FP32, or I32 words in logical coordinate order;
- `row-split-k128-v1` yields the grouped signed codes and binary16 scales for logical columns
  `0..K-1`, discarding physical columns `K..K_pad-1`;
- `raw-bytes-v1` yields the enclosing resource bytes.

Dequantized values follow the reconstruction rule in `ninfer-tensor-formats.md`. This document does
not select a quantization encoder, output dtype, accumulation dtype, kernel, runtime device layout,
or model consumer.
