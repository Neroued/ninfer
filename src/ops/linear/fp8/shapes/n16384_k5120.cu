#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 5120>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using C2       = Fp8A16SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::SharedPhase,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4       = Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

using C8     = Fp8A16SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C10    = Fp8A16SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2>;
using Full10 = Fp8A16SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 2) return fp8_linear_a16_simt<Geometry, 2, C2>(x, weight, out, stream);
    if (tokens <= 4) return fp8_linear_a16_simt<Geometry, 4, C4>(x, weight, out, stream);
    if (tokens == 10)
        return fp8_linear_a16_simt<Geometry, 10, Full10, true>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, C8>(x, weight, out, stream);
    if (tokens <= 10) return fp8_linear_a16_simt<Geometry, 10, C10>(x, weight, out, stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R64K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    launch_fp8_a8<Geometry, Fp8A8T64R128K128>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 11; }
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384, 5120, launch_a16, launch_a8, uses_a8};
} // namespace ninfer::ops::detail
