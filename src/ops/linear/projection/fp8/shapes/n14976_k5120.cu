#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<14976, 5120>;

// Retain the complete shape in CUDA entry names. K alone collides with other
// translation units that instantiate an identical schedule for a different N.
template <class Schedule, class Shape = Geometry>
struct ShapeSchedule : Schedule {};

using Gemv    = ShapeSchedule<Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>;
using Hot     = ShapeSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>;
using T256    = ShapeSchedule<Fp8A8TmaMmaSchedule<256, 128, 128, 2, 4, 2, 1>>;
using Bulk128 = ShapeSchedule<
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 4>>;
using Bulk256 = ShapeSchedule<Fp8A8TmaSplitKSchedule<T256, 170, 4, 4>>;

using T32R32 = ShapeSchedule<Fp8A8T32R32K128>;

void launch_a16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, w, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, ShapeSchedule<Fp8SlicedInstance<16, 4, 1>>>(
            x, w, out, stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, ShapeSchedule<Fp8SlicedInstance<32, 4, 2>>>(
            x, w, out, stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, ShapeSchedule<Fp8SlicedInstance<32, 4, 1>>>(
            x, w, out, stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry,
                                  ShapeSchedule<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>>(
            x, w, out, stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry,
                                  ShapeSchedule<Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>>(
            x, w, out, stream);
    fp8_linear_a16_mma<Geometry, ShapeSchedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>(
        x, w, out, stream);
}

void launch_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32) return launch_fp8_a8<Geometry, T32R32>(x, w, out, scratch, stream);
    const int tokens = x.ne[1];
    if (tokens <= 128) return launch_fp8_a8_tma<Geometry, Hot>(x, w, out, scratch, stream);
    if (tokens <= 256) return launch_fp8_a8_tma<Geometry, T256>(x, w, out, scratch, stream);
    if (tokens <= 896 || (tokens >= 1025 && tokens <= 1408))
        return launch_fp8_a8_tma<Geometry, Bulk128>(x, w, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk256>(x, w, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // A monotone bound also covers intervals whose final point does not split K.
    if (max_tokens >= 897) return Bulk256::kPartialBytes;
    return max_tokens >= 257 ? Bulk128::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N14976K5120{14976,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
