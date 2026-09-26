#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<34816, 5120>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using C2       = Fp8A16SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::SharedPhase,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4       = Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full4    = Fp8A16SimtSchedule<8, 2, 8, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 4) return fp8_linear_a16_simt<Geometry, 4, Full4, true>(x, weight, out, stream);
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 2) return fp8_linear_a16_simt<Geometry, 2, C2>(x, weight, out, stream);
    if (tokens <= 4) return fp8_linear_a16_simt<Geometry, 4, C4>(x, weight, out, stream);
    if (tokens <= 8)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 4, 1>>(x, weight, out,
                                                                             stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 2, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 16)
        return launch_fp8_a8<Geometry, Fp8A8T16R128K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R128K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 96)
        return launch_fp8_a8<Geometry, Fp8A8T32R128K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128>(x, weight, out, scratch, stream);
    launch_fp8_a8<Geometry, Fp8A8T64R128K128>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t min_tokens, std::int32_t max_tokens) {
    return min_tokens == 1 || max_tokens >= 5;
}
} // namespace

const Fp8LinearShape kFp8N34816K5120{34816, 5120, launch_a16, launch_a8, uses_a8};
} // namespace ninfer::ops::detail
