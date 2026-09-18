#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<34816, 5120>;
using Gemv      = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using A8        = Fp8A8DefaultSchedule;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using FullChunk = Fp8SimtSchedule<8, 2, 8, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv>;
    if (tokens == 4) return launch_fp8_simt<Geometry, 4, FullChunk, true>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

bool uses_a8(std::int32_t min_tokens, std::int32_t max_tokens) {
    return min_tokens == 1 || max_tokens >= 5;
}
} // namespace

const Fp8LinearShape kFp8N34816K5120{34816, 5120, launch_fp8_a16_chunks<4, select_a16>,
                                     launch_fp8_a8<Geometry, A8>, uses_a8};
} // namespace ninfer::ops::detail
