# QUS Weight File Format v1

This is the binary contract between the offline Python packer and the C++ `WeightStore` loader.
All integers are little-endian. All reserved bytes must be zero. Payload offsets are absolute file
offsets. Payload blobs must be 256-byte aligned and non-overlapping.

## Header

The header is exactly 256 bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic bytes `QUSWGT01` |
| 8 | 4 | `version = 1` |
| 12 | 4 | `endian_tag = 0x01020304` |
| 16 | 4 | `header_size = 256` |
| 20 | 4 | `tensor_entry_size = 256` |
| 24 | 4 | `tensor_count` |
| 28 | 4 | reserved zero |
| 32 | 8 | `tensor_table_offset` |
| 40 | 8 | `payload_offset` |
| 48 | 8 | `file_size` |
| 56 | 64 | NUL-terminated UTF-8 `model_id` |
| 120 | 4 | `hidden_size` |
| 124 | 4 | `intermediate_size` |
| 128 | 4 | `num_layers` |
| 132 | 4 | `full_attention_layers` |
| 136 | 4 | `gdn_layers` |
| 140 | 4 | `attention_heads` |
| 144 | 4 | `kv_heads` |
| 148 | 4 | `head_dim` |
| 152 | 4 | `gdn_key_heads` |
| 156 | 4 | `gdn_value_heads` |
| 160 | 4 | `gdn_value_head_dim` |
| 164 | 4 | `gdn_key_head_dim` |
| 168 | 4 | `gdn_conv_width` |
| 172 | 4 | `vocab_size` |
| 176 | 4 | `max_position_embeddings` |
| 180 | 76 | reserved zero |

`tensor_table_offset` must be at least `header_size`. The byte range
`[tensor_table_offset, tensor_table_offset + tensor_count * tensor_entry_size)` must end at or
before `payload_offset`.

## Tensor Entry

Each tensor table entry is exactly 256 bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 96 | NUL-terminated UTF-8 `name` |
| 96 | 48 | NUL-terminated UTF-8 `role` |
| 144 | 4 | signed `layer`, `-1` for global |
| 148 | 1 | `kind`: `0=dense`, `1=quant_w4` |
| 149 | 1 | `dtype`: `0=BF16`, `1=FP32`, `2=I32`, `3=U8` |
| 150 | 1 | `quant_layout`: `0=W4A16KernelPackedV1`, `255=none` |
| 151 | 1 | `rank`, 1 to 4 |
| 152 | 16 | `shape[4]`, `uint32`, trailing dims must be 1 |
| 168 | 8 | absolute `data_offset` |
| 176 | 8 | `data_nbytes` |
| 184 | 1 | `scale_dtype`: `0=BF16`, `1=FP32`, `2=I32`, `3=U8`, `255=none` |
| 185 | 1 | `scale_rank`, 0 when absent |
| 186 | 2 | reserved zero |
| 188 | 16 | `scale_shape[4]`, `uint32`, zeros when absent |
| 204 | 8 | absolute `scale_offset`, 0 when absent |
| 212 | 8 | `scale_nbytes`, 0 when absent |
| 220 | 4 | logical `n`, 0 for dense |
| 224 | 4 | logical `k`, 0 for dense |
| 228 | 4 | quant `group`, 0 for dense |
| 232 | 24 | reserved zero |

Dense entries require `quant_layout = 255`, `scale_dtype = 255`, `n = k = group = 0`, no scale
payload, and `data_nbytes == product(shape[0:rank]) * dtype_size(dtype)`.

Quant W4 entries require `kind = 1`, `dtype = U8`, `quant_layout = W4A16KernelPackedV1`, positive
`n`, `k`, and `group`, and nonzero qdata and scale payloads. The v1 packed layout is row-major
logical `[N,K]`, two 4-bit weights per byte:

```
data_nbytes = n * ceil(k / 2)
scale_dtype = FP32
scale_rank = 2
scale_shape = { n, ceil(k / group), 1, 1 }
scale_nbytes = n * ceil(k / group) * sizeof(float)
```

## Payload Validation

Every nonzero payload range must satisfy:

- `offset >= payload_offset`
- `offset % 256 == 0`
- `nbytes > 0`
- `offset + nbytes <= file_size`
- no overlap with any other payload range

The file is malformed if fixed strings lack a NUL terminator, reserved fields are nonzero,
table/header ranges overlap the payload region, enum tags are unknown, ranks are invalid, shape
metadata does not match byte counts, or duplicate `(kind, role, layer)` entries appear.
