#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

// `stacked` admits a plane assembled from several separately quantised matrices, which carries a
// divisor per source and is read per row. Dense routes leave it false: their kernels read the one
// scalar, so a plane with more than one divisor is not something they can execute.
Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation,
                                          bool stacked = false);

} // namespace ninfer::ops::detail
