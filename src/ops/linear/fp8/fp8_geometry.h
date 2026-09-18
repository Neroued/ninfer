#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

using Fp8N14336K5120             = Fp8Geometry<14336, 5120>;
using Fp8N16384K5120             = Fp8Geometry<16384, 5120>;
using Fp8N34816K5120             = Fp8Geometry<34816, 5120>;
using Fp8N248320K5120            = Fp8Geometry<248320, 5120>;
using Fp8N5120K6144              = Fp8Geometry<5120, 6144>;
using Fp8N5120K17408             = Fp8Geometry<5120, 17408>;
using Fp8Activation5120Geometry  = Fp8ActivationGeometry<5120>;
using Fp8Activation6144Geometry  = Fp8ActivationGeometry<6144>;
using Fp8Activation17408Geometry = Fp8ActivationGeometry<17408>;

enum class Fp8GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N248320K5120,
    N5120K6144,
    N5120K17408,
};

inline Fp8GeometryId resolve_fp8_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8N14336K5120::kOutputRows && input_rows == Fp8N14336K5120::kInputRows) {
        return Fp8GeometryId::N14336K5120;
    }
    if (output_rows == Fp8N16384K5120::kOutputRows && input_rows == Fp8N16384K5120::kInputRows) {
        return Fp8GeometryId::N16384K5120;
    }
    if (output_rows == Fp8N34816K5120::kOutputRows && input_rows == Fp8N34816K5120::kInputRows) {
        return Fp8GeometryId::N34816K5120;
    }
    if (output_rows == Fp8N248320K5120::kOutputRows && input_rows == Fp8N248320K5120::kInputRows) {
        return Fp8GeometryId::N248320K5120;
    }
    if (output_rows == Fp8N5120K6144::kOutputRows && input_rows == Fp8N5120K6144::kInputRows) {
        return Fp8GeometryId::N5120K6144;
    }
    if (output_rows == Fp8N5120K17408::kOutputRows && input_rows == Fp8N5120K17408::kInputRows) {
        return Fp8GeometryId::N5120K17408;
    }
    throw std::invalid_argument("unsupported FP8 problem");
}

} // namespace ninfer::ops::detail
