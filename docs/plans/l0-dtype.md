# L0 dtype Plan

## Scope

Implement the model-agnostic dtype primitive from `docs/l0-infrastructure-design.md` section 4.
This component has no CUDA ownership, no device memory ownership, and no dependency on model
dimensions.

## Types and API

Modify `include/qus/core/dtype.h`:

```cpp
#include <cstddef>
#include <cstdint>

namespace qus {

enum class DType : std::uint8_t {
  BF16 = 0,
  FP32 = 1,
  I32 = 2,
  U8 = 3,
};

std::size_t dtype_size(DType dtype);

}  // namespace qus
```

Modify `src/core/dtype.cpp`:

- Include `<stdexcept>` for invalid serialized enum values.
- Implement `dtype_size(DType)` with the exact byte sizes from the design:
  - `BF16` = 2 bytes
  - `FP32` = 4 bytes
  - `I32` = 4 bytes
  - `U8` = 1 byte, representing packed 4-bit storage bytes
- Throw `std::invalid_argument` for any invalid enum value so malformed metadata fails before
  allocation math silently proceeds.

## Memory and Ownership

`DType` owns no memory. `dtype_size` performs no allocation and has no CUDA dependency.

## Edge Cases

- Invalid enum values created by casting from serialized metadata must fail with
  `std::invalid_argument`.
- `U8` returns 1 byte because it is the storage byte for packed 4-bit weights, not one logical
  4-bit element.

## Tests

Create `tests/test_dtype.cpp` with a host test that returns nonzero on runtime failures,
including in Release builds, and checks:

- `DType` uses the `std::uint8_t` ABI/storage contract.
- All four supported dtype sizes.
- Invalid cast value throws `std::invalid_argument`.

Update `CMakeLists.txt` to enable CTest and build `qus_l0_tests`, starting with
`tests/test_dtype.cpp`. The test executable links `qus_core`.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
