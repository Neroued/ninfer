#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
