# NInfer Persistent Storage Layouts

This reference records the persistent tensor layouts and resource encoding used by current
`.ninfer` artifacts, including alignment, byte order, padding, encoded-size rules, and logical
decode. Numeric semantics come from [`tensor-formats.md`](tensor-formats.md); framing and object
ranges come from [`artifact-container.md`](artifact-container.md).

## 1. Registered identities

The storage registry contains exactly these identities:

| Identity | Kind | Compatible numeric formats | Logical shape | Object alignment |
|---|---|---|---|---:|
| `contiguous_le_v1` | tensor layout | `bf16`, `fp32`, `int32` | rank `0..16` | 256 bytes |
| `row_split_k128_v1` | tensor layout | `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | rank 2 `[N,K]` | 256 bytes |
| `block_scale_k16_m128x4_v1` | tensor layout | `nvfp4` | rank 2 `[N,K]`, `N % 128 == 0`, `K % 64 == 0` | 256 bytes |
| `batched_block_scale_k16_m128x4_v1` | tensor layout | `nvfp4` | rank 3 `[E,N,K]`, `N % 128 == 0`, `K % 64 == 0` | 256 bytes |
| `block_scale_m128_k128_v1` | tensor layout | `fp8_e4m3fn_block128x128_bf16` | `[N,K]` or `[E,N,K]`, N and K multiples of 128 | 256 bytes |
| `row_scale_v1` | tensor layout | `fp8_e4m3fn_row_bf16` | rank 2 `[N,K]` | 256 bytes |
| `row_paged_g16_4k_v1` | tensor layout | `nf4_g16_e4m3fn` | rank 2 `[N,K]`, `K % 16 == 0`, encoded row ≤ 4096 bytes | 4096 bytes |
| `raw_bytes_v1` | resource encoding | not applicable | nonempty byte string | 1 byte |

These format/layout pairs define the current codec support. Native consumer requirements are
covered separately in Section 9.

Object alignment applies to the object's payload-relative `offset` in the `.ninfer` JSON. Internal
plane offsets and padding belong to the selected layout. Inter-object padding belongs to the
container and is not included in an object's `bytes`.

The helper used below is:

```text
align_up(x, a) = ceil_div(x, a) * a
```

All formula inputs and intermediate results are nonnegative integers. The container implementation
must reject a shape or calculation that cannot be represented by its file-offset and size types.

## 2. `contiguous_le_v1`

### 2.1 Logical traversal

`contiguous_le_v1` stores direct logical words in C order: the last logical dimension varies
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
| `bf16` | 2 | the exact 16-bit bfloat16 logical word, little-endian |
| `fp32` | 4 | the exact 32-bit IEEE-754 binary32 logical word, little-endian |
| `int32` | 4 | the exact 32-bit two's-complement logical word, little-endian |

Signed zero, subnormal, infinity, NaN payload, and integer-word behavior are determined by the
numeric-format contract. The layout only preserves the word bits.

### 2.3 Encoded size

There is no internal prefix, stride table, per-row padding, or trailing padding:

```text
payload_bytes = elements * bytes_per_element(format)
```

The tensor object's JSON `bytes` must equal `payload_bytes` exactly.

## 3. `row_split_k128_v1`

### 3.1 Logical and physical geometry

The logical tensor is a positive rank-two matrix `[N,K]`. The format supplies code width `b` and
group size `G`:

| Format | `b` | `G` | Base bytes per group `B` | High bytes per group `H` |
|---|---:|---:|---:|---:|
| `q4_g64_fp16` | 4 | 64 | 32 | 0 |
| `q5_g64_fp16` | 5 | 64 | 32 | 8 |
| `q6_g64_fp16` | 6 | 64 | 32 | 16 |
| `q8_g32_fp16` | 8 | 32 | 32 | 0 |

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

Q4 and Q8 have no high-bit bytes. They still place the scale plane at the first 256-byte boundary
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

For Q8, each lane occupies one byte containing its exact 8-bit two's-complement word. Lane `i`
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
`N=row_count`. This produces another valid `row_split_k128_v1` tensor without decoding or repacking
individual codes.

## 4. `block_scale_k16_m128x4_v1`

This layout stores only rank-two `nvfp4` matrices `[N,K]` satisfying:

```text
N > 0
K > 0
N % 128 == 0
K % 64 == 0
```

It adds no logical padding. Let:

```text
code_plane_bytes      = N * K / 2
scale_plane_offset    = align_up(code_plane_bytes, 256)
scale_plane_bytes     = N * K / 16
weight_divisor_offset = scale_plane_offset + scale_plane_bytes
payload_bytes         = weight_divisor_offset + 4
```

The payload is a row-major E2M1 packed-code plane, zero padding to `scale_plane_offset`, a
swizzled E4M3FN scale plane, and the little-endian FP32 weight-divisor word. Within each packed code
byte, the low nibble is the smaller K coordinate and the high nibble is the next coordinate.

For logical row `n`, scale-group coordinate `g=floor(k/16)`, and `K_tiles=K/64`, define:

```text
row_tile   = floor(n / 128)
row_inner  = n % 128
scale_tile = floor(g / 4)
scale_lane = g % 4
```

The scale word's byte offset within the scale plane is:

```text
(row_tile * K_tiles + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

Layout decoding must recover the original packed E2M1 words, natural `[N,K/16]` E4M3FN scale-word
matrix, and exact divisor word. It never decodes and re-encodes either floating-point format.

### 4.1 `batched_block_scale_k16_m128x4_v1`

The bank layout stores `[E,N,K]`, E positive, with the same N/K constraints and inner-matrix
code packing and scale swizzle as Section 4. It has no per-matrix padding:

```text
code_matrix_bytes     = N*K/2
scale_matrix_bytes    = N*K/16
code_plane_bytes      = E*code_matrix_bytes
scale_plane_offset    = align_up(code_plane_bytes, 256)
scale_plane_bytes     = E*scale_matrix_bytes
weight_divisor_offset = scale_plane_offset + scale_plane_bytes
payload_bytes         = weight_divisor_offset + 4*E
```

The code plane contains all matrices in C order. Matrix e's scale bytes start at
`scale_plane_offset + e*scale_matrix_bytes`; its FP32 divisor starts at
`weight_divisor_offset + 4*e`. All FP32 words are little-endian. Any gap before the scale plane
is zero. A matrix slice keeps its own divisor; flattening the bank into a single scalar-scaled
matrix is not a valid reinterpretation.

## 5. `row_scale_v1`

`row_scale_v1` stores only rank-two `fp8_e4m3fn_row_bf16` matrices `[N,K]` with positive
dimensions. It adds no logical or physical matrix padding. Let:

```text
code_plane_bytes   = N * K
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes  = N * 2
payload_bytes      = scale_plane_offset + scale_plane_bytes
```

The payload begins with one row-major E4M3FN byte per logical weight. Coordinate `[n,k]` is stored
at code-plane offset `n * K + k`. Zero bytes fill the interval from `code_plane_bytes` to
`scale_plane_offset`.

The scale plane contains one little-endian BF16 word per logical row in increasing `n` order. Scale
word `n` begins at `scale_plane_offset + 2 * n`. The layout neither converts the BF16 multiplier nor
combines it with its E4M3FN row. Code and scale validity and represented-weight reconstruction are
defined by `fp8_e4m3fn_row_bf16` in [`tensor-formats.md`](tensor-formats.md).

A logical row view consists of its `K` consecutive code bytes and its one BF16 scale word; those two
spans are not one assumed-contiguous payload range. A standalone consecutive slice or row gather is
encoded by concatenating the selected code rows, recomputing the scale-plane alignment for the new
row count, and appending the selected scale words in the same row order. It does not decode or
requantize either plane.

### 5.1 `block_scale_m128_k128_v1`

This layout stores `fp8_e4m3fn_block128x128_bf16` as a rank-two matrix or rank-three bank.
N and K are positive multiples of 128. For rank two use E=1 in these byte-size formulas:

```text
code_plane_bytes  = E*N*K
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes = E*(N/128)*(K/128)*2
payload_bytes     = scale_plane_offset + scale_plane_bytes
```

Codes use natural C-order `[E,N,K]`. Scales use natural C-order `[E,N/128,K/128]`, stored as
little-endian BF16 words without swizzling or per-matrix padding. Scale words require two-byte
alignment; NVFP4 tile-alignment rules do not apply. Any gap before the scale plane is zero.
Gate/up can concatenate along N when their boundary is a multiple of 128, retaining every
block's independent multiplier.

## 6. `row_paged_g16_4k_v1`

This independent layout stores NF4 codes and G16 scales together in each row. It has positive
rank-two shape `[N,K]`, `K % 16 == 0`, and `row_bytes <= 4096`:

```text
row_bytes             = K/2 + K/16
rows_per_page         = floor(4096 / row_bytes)
page_count            = ceil_div(N, rows_per_page)
row_offset(r)         = floor(r / rows_per_page)*4096 + (r % rows_per_page)*row_bytes
weight_divisor_offset = page_count*4096
payload_bytes         = weight_divisor_offset + 4
```

Each row begins with `K/2` packed-code bytes (low nibble is the even K coordinate), followed by
`K/16` E4M3FN scale bytes in increasing group order. Rows never cross pages. Page tails and
unused rows in the final page contain zero bytes. The little-endian FP32 divisor follows the
complete final page. The tensor object's payload-relative offset is aligned to 4096 bytes.

For `[2500012,160]`, a row occupies 90 bytes, a page holds 45 rows, and 55556 pages plus the
divisor occupy 227557380 bytes. The final page holds 37 valid rows. A logical zero NF4 group
uses packed `0x77` codes and scale zero; page padding has no logical values.

`weight_paged_row_offset` and `weight_scale_offset` expose parent-relative C++ addresses;
`row_paged_geometry` provides the corresponding Python geometry. Paged rows cannot be exposed
as constant-stride code/scale planes or passed to the current native Linear `Weight` bridge.

## 7. `raw_bytes_v1`

`raw_bytes_v1` is a resource encoding, not a tensor layout. Its enclosing object payload is
the resource byte string itself:

```text
payload_bytes = resource_length
alignment     = 1
```

It has no embedded length, header, terminator, filename, character encoding, compression, or
trailing padding. The resource object's JSON `bytes` is its exact nonzero length, and a reader
returns the complete span unchanged. A model contract assigns a resource name and interprets those
bytes; the common encoding does not infer that meaning from the name.

## 8. Decode boundary

Layout decoding yields only persistent logical words:

- `contiguous_le_v1` yields the direct BF16, FP32, or I32 words in logical coordinate order;
- `row_split_k128_v1` yields the grouped signed codes and binary16 scales for logical columns
  `0..K-1`, discarding physical columns `K..K_pad-1`;
- `block_scale_k16_m128x4_v1` yields the packed E2M1 words, natural E4M3FN group-scale words, and
  matrix-level FP32 weight divisor;
- `batched_block_scale_k16_m128x4_v1` yields those words for each matrix and its independent divisor;
- `block_scale_m128_k128_v1` yields natural E4M3FN codes and BF16 128×128 block multipliers;
- `row_scale_v1` yields the natural row-major E4M3FN code words and one BF16 multiplier per logical
  row;
- `row_paged_g16_4k_v1` yields packed NF4 codes, natural G16 E4M3FN scales and the object FP32 divisor;
- `raw_bytes_v1` yields the enclosing resource bytes.

Dequantized values follow the reconstruction rule in `tensor-formats.md`. This document does
not select a quantization encoder, output dtype, accumulation dtype, kernel, runtime device layout,
or model consumer.

## 9. Logical views and native operands

Bindings address C-order logical element ranges of a parent object. The parent retains its full
geometry and backing allocation, so a view can locate code and scale planes using the original
matrix dimensions. Materialization uploads each required parent once and binds non-owning views.

[`weight_view.cpp`](../../src/core/weight_view.cpp) provides plane addressing and the bridge to
native operands. Direct tensors can use a contiguous element range. Grouped integer matrices can
use consecutive complete rows with unchanged K, using independent code, high-bit and scale pointers.

The native `Weight` bridge accepts consecutive complete rows with unchanged K for row-scaled FP8.
Its code and scale pointers address the original parent planes independently; logical N does not
determine a borrowed view's physical scale-plane offset. NVFP4 accepts complete
128-row tiles within one matrix divisor domain, including within a bank. Its native code and scale
pointers are rebased independently; the inner layout remains `BlockScaleK16M128x4`. Block-scaled
FP8 accepts 128-row-aligned views with unchanged K and preserves the two-dimensional scale grid.
Its native scale axes are `[K/128,N/128]` with byte strides `[2,2*(K/128)]`.
`payload` and `payload_bytes` always describe the original backing, not a fabricated contiguous
slice. Actual Ops still decide whether they support the format and geometry.

Materialized parents borrow a Host divisor span from their backing owner. A bank has E words;
a single NVFP4 matrix or NF4 parent has one. Preparing a CUDA operand reads this Host metadata
and never dereferences its Device payload on the CPU.

Offline codecs can produce a standalone slice with its own plane offsets. The loader does not
perform that transformation. An execution implementation that accepts additional view forms must
consume the original parent geometry correctly.
