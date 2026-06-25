# L0 weight_store Plan

## Scope

Define the fixed weight-file contract in `tools/weight_format.md`, then implement
`WeightStore`/loader from `docs/l0-infrastructure-design.md` section 5.2 and `docs/design.md`
section 10. This component reads the one fixed binary file, validates metadata against
caller-provided expected model metadata, uploads payload blobs into the Weights `DeviceArena`
through pinned host staging, and exposes dense `Tensor` and W4 `QuantWeight` descriptors.

This component does not implement quantization, packing, relayout, Python tooling, safetensors,
runtime dequantization, model-card logic, or any L1 kernels.

## File Format Contract

Replace the stub `tools/weight_format.md` with a v1 binary format:

- Fixed little-endian file.
- Header magic `QUSWGT01`, version `1`, endian tag `0x01020304`, header size, tensor-entry size,
  tensor count, absolute tensor-table offset, absolute payload offset, file size, model id, and
  model dims. The doc must include a byte-offset table with exact field order, widths, numeric enum
  values, 64-byte `model_id`, and reserved fields that must be zero.
- Model dims include: hidden size, intermediate size, layer count, full-attention layer count,
  GDN layer count, attention query heads, KV heads, attention head dim, GDN key heads, GDN value
  heads, GDN value head dim (`dv`), GDN key head dim (`dk`), GDN conv width, vocab size, and max
  position embeddings. L0 treats `0` in expectations as “do not validate this field”; L2 passes
  concrete values.
- Tensor table entry fields include byte-exact order and widths: 96-byte name, 48-byte role, layer
  index (`-1` for global), kind (`dense` or `quant_w4`), dtype, quant layout, rank/shape, absolute
  data offset/bytes, scale dtype/rank/shape/offset/bytes, logical `n`, `k`, quant group, and
  reserved-zero fields.
- Payload offsets are absolute file offsets, must be within file size, 256-byte aligned, nonzero
  byte counts, `>= payload_offset`, and non-overlapping. Header/table ranges must not overlap the
  payload region.
- Dense tensor entries require `data_nbytes == Tensor(nullptr, dtype, shape).bytes()`, no scale
  payload, and expose `WeightStore::weight(role, layer)`.
- Quant entries require `dtype == U8`, `layout == W4A16KernelPackedV1`, positive `n/k/group`,
  nonzero qdata and scale payloads, and expose `WeightStore::qweight(role, layer)`. For v1
  `W4A16KernelPackedV1`, qdata is row-major logical `[N,K]` with two 4-bit weights per byte, so
  `data_nbytes == n * ceil(k/2)`. Scales are FP32 with shape `{n, ceil(k/group)}` and byte count
  exactly matching that derived shape.

## Types and API

Modify `include/qus/core/weight_store.h`:

```cpp
#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qus {

enum class WeightEntryKind : std::uint8_t {
  Dense = 0,
  QuantW4 = 1,
};

struct WeightFileExpectations {
  std::string model_id;
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t num_layers = 0;
  std::uint32_t full_attention_layers = 0;
  std::uint32_t gdn_layers = 0;
  std::uint32_t attention_heads = 0;
  std::uint32_t kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t gdn_key_heads = 0;
  std::uint32_t gdn_value_heads = 0;
  std::uint32_t gdn_value_head_dim = 0;
  std::uint32_t gdn_key_head_dim = 0;
  std::uint32_t gdn_conv_width = 0;
  std::uint32_t vocab_size = 0;
  std::uint32_t max_position_embeddings = 0;
};

class WeightStore {
public:
  explicit WeightStore(WeightFileExpectations expected = {});

  void load(const char* path, DeviceArena& weights, DeviceContext& ctx);
  QuantWeight qweight(std::string_view role, int layer) const;
  Tensor weight(std::string_view role, int layer) const;
  std::size_t dense_count() const noexcept;
  std::size_t quant_count() const noexcept;
  void clear() noexcept;

private:
  // implementation-owned descriptor records
};

}  // namespace qus
```

Implement in `src/core/weight_store.cpp`:

- Parse header and fixed-size table with `std::ifstream` in binary mode before reading any payload
  bytes.
- Validate magic/version/endian/header/table offsets/file size/table size.
- Validate each expected model field when the expectation value is nonzero; validate model id when
  the expected string is non-empty.
- Validate all tensor entries before allocating or uploading. This includes shape ranks, dtype
  tags, byte counts, 256-byte alignment, payload ranges, and non-overlap.
- Require fixed string fields to contain at least one NUL terminator and reject duplicate
  `(kind, role, layer)` entries.
- Preflight total Weights arena space before allocation with conservative 256-byte alignment
  overhead per dense data, quant qdata, and quant scales payload so malformed/oversized files fail
  before consuming the monotonic arena.
- Convert fixed table shapes to existing arena APIs with a small rank-switch helper for ranks 1-4.
  Do not pass dynamic arrays directly to `DeviceArena::alloc`.
- Allocate dense payloads via the rank-switch helper.
- Allocate quant qdata as `weights.alloc(DType::U8, {static_cast<int32_t>(data_nbytes)})` after
  validating `data_nbytes <= INT32_MAX`, and scale payloads via the rank-switch helper.
- Stream each payload range from the file through a bounded `PinnedHostBuffer` staging buffer
  (1 MiB maximum chunk) and `cudaMemcpyAsync` on `ctx.load_stream`. Synchronize the load stream
  before reusing or destroying the staging buffer so async H2D never observes freed pinned memory.
- Store dense records and quant records in vectors with role/name/layer metadata.
- `weight(role, layer)` and `qweight(role, layer)` perform exact role/layer lookup and throw
  `std::out_of_range` if missing or wrong kind.
- `clear()` drops descriptor records but does not free arena memory.

Update `QuantWeight` in `include/qus/core/tensor.h` to carry the scale descriptor metadata needed
by the loader and future kernels:

```cpp
const void* scales;
DType scale_dtype;
std::int32_t scale_ne[4];
std::int64_t scale_nb[4];
```

Default scale dtype is `DType::FP32`, scale dims default to `{1,1,1,1}`, and default strides are
zero until populated by the loader.

## Memory and Ownership

`WeightStore` owns host descriptor vectors only. Device payloads are owned by the Weights
`DeviceArena`. Returned `Tensor` and `QuantWeight` objects are non-owning device views. The loader
does not allocate during inference steps.

## Error Handling

Throw `std::runtime_error` for file open/read errors, CUDA allocation/upload failures, malformed
headers, unsupported version, bad magic, endian mismatch, model metadata mismatch, invalid tensor
metadata, overlapping payloads, and missing required scale payloads. Throw `std::bad_alloc` before
arena consumption when preflight detects insufficient remaining Weights capacity. Throw
`std::out_of_range` for missing lookups. Direct-check `cudaMemcpyAsync` and
`cudaStreamSynchronize` results and throw `std::runtime_error`; do not use `CUDA_CHECK` in the
loader upload path because malformed files and upload failures should be testable exceptions.

## Edge Cases

- Strings in the binary header/table are fixed-size UTF-8 byte fields and must contain a NUL byte;
  unterminated fields are malformed.
- Payload ranges are absolute file offsets, so table and payload validation does not depend on
  current stream positions.
- Duplicate `(kind, role, layer)` entries are rejected.
- `layer == -1` is allowed for global tensors; any other negative layer is invalid.
- Dense `U8` tensors are allowed for non-quant byte payloads, but `QuantWeight` entries must use
  `WeightEntryKind::QuantW4`.

## Tests

Create `tests/test_weight_store.cpp` as a standalone CUDA smoke test executable with explicit
nonzero failure returns:

- Skip cleanly if CUDA reports no usable device; fail on unexpected CUDA errors.
- Write a tiny valid v1 weight file at test runtime with matching expectations:
  - Dense global role `norm`, layer `-1`, dtype `FP32`, shape `{4}`, payload four floats.
  - Quant role `q_proj`, layer `0`, qdata `U8` 8 bytes, scales `FP32` shape `{2,2}`, `n=2`,
    `k=8`, `group=4`, layout `W4A16KernelPackedV1`.
- Load it with `WeightStore expected`, `DeviceArena weights`, and `DeviceContext`.
- Verify dense and quant counts, dense tensor shape/dtype and payload bytes copied back from
  device, quant descriptor fields, qdata bytes, and scale bytes copied back from device.
- Verify missing dense/quant lookups throw `std::out_of_range`.
- Verify malformed files throw before consuming the arena:
  - bad magic
  - invalid version
  - invalid endian tag
  - hidden-size expectation mismatch
  - invalid dtype, invalid layout, invalid rank
  - unterminated name or role
  - unaligned offset, out-of-range offset, payload before `payload_offset`
  - table/header range overlapping the payload region
  - duplicate `(kind, role, layer)`
  - overlapping payload ranges
  - quant entry missing scales
  - quant qdata byte count mismatch
  - quant serialized shape byte count mismatch
  - quant scale shape or byte count mismatch
  - dense byte count not matching shape
- Verify undersized Weights arena throws `std::bad_alloc` before changing `used()`.

Update `CMakeLists.txt` with a separate `qus_weight_store_test` executable and CTest entry.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
