#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<5120, 6144>;
using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using A8       = Fp8A8DefaultSchedule;
using C2       = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4       = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8       = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C12      = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C16      = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C24      = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                 Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using FullChunk =
    Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                    Fp8SimtBlockOrder::TokenTilesContiguous, 1>;
using Full17 = Fp8SimtSchedule<8, 2, 16, 17, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full18 = Fp8SimtSchedule<8, 2, 16, 18, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full19 = Fp8SimtSchedule<8, 2, 16, 19, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full20 = Fp8SimtSchedule<8, 2, 8, 20, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv>;
    if (tokens == 24) return launch_fp8_simt<Geometry, 24, FullChunk, true>;
    // These short SIMT extents retain a measured advantage over padded capacities.
    if (tokens == 17) return launch_fp8_simt<Geometry, 17, Full17, true>;
    if (tokens == 18) return launch_fp8_simt<Geometry, 18, Full18, true>;
    if (tokens == 19) return launch_fp8_simt<Geometry, 19, Full19, true>;
    if (tokens == 20) return launch_fp8_simt<Geometry, 20, Full20, true>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8>;
    if (tokens <= 12) return launch_fp8_simt<Geometry, 12, C12>;
    if (tokens <= 16) return launch_fp8_simt<Geometry, 16, C16>;
    if (tokens <= 24) return launch_fp8_simt<Geometry, 24, C24>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 25; }
} // namespace

const Fp8LinearShape kFp8N5120K6144{5120, 6144, launch_fp8_a16_chunks<24, select_a16>,
                                    launch_fp8_a8<Geometry, A8>, uses_a8};
} // namespace ninfer::ops::detail
