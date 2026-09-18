#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

using Nvfp4N14336K5120 = Nvfp4Geometry<14336, 5120>;
using Nvfp4N16384K5120 = Nvfp4Geometry<16384, 5120>;
using Nvfp4N34816K5120 = Nvfp4Geometry<34816, 5120>;
using Nvfp4N5120K6144  = Nvfp4Geometry<5120, 6144>;
using Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

enum class Nvfp4GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N5120K6144,
    N5120K17408,
};

inline Nvfp4GeometryId resolve_nvfp4_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4N14336K5120::kOutputRows &&
        input_rows == Nvfp4N14336K5120::kInputRows) {
        return Nvfp4GeometryId::N14336K5120;
    }
    if (output_rows == Nvfp4N16384K5120::kOutputRows &&
        input_rows == Nvfp4N16384K5120::kInputRows) {
        return Nvfp4GeometryId::N16384K5120;
    }
    if (output_rows == Nvfp4N34816K5120::kOutputRows &&
        input_rows == Nvfp4N34816K5120::kInputRows) {
        return Nvfp4GeometryId::N34816K5120;
    }
    if (output_rows == Nvfp4N5120K6144::kOutputRows && input_rows == Nvfp4N5120K6144::kInputRows) {
        return Nvfp4GeometryId::N5120K6144;
    }
    if (output_rows == Nvfp4N5120K17408::kOutputRows &&
        input_rows == Nvfp4N5120K17408::kInputRows) {
        return Nvfp4GeometryId::N5120K17408;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

} // namespace ninfer::ops::detail
