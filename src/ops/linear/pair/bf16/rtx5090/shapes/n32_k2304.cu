#include "ops/linear/pair/dispatch.h"
#include "ops/linear/pair/bf16/rtx5090/small_launch.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n32_k2304.cuh"

namespace ninfer::ops::detail {
namespace {

using bf16_n32_k2304::StaticK;
using Sliced8 = StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 6, 64, 2, Cache::cg, Cache::cg, 1, 1>>;
using Mma32 =
    StaticK<Bf16A16MmaSchedule<16, 32, 256, 16, 8, 2, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma64 = StaticK<Bf16A16TmaMmaSchedule<16, 64, 128, 16, 16, 2>>;

using Launch = void (*)(Bf16A16Operands, Bf16A16Operands, LinearBf16Output, LinearBf16Output,
                        cudaStream_t);

Launch select(int tokens) {
    if (tokens <= 8) return launch_bf16_pair_simt<bf16_n32_k2304::DecodeSimt>;
    if (tokens <= 384) return launch_bf16_pair_sliced<Sliced8>;
    if (tokens <= 768) return launch_bf16_pair_sliced<bf16_n32_k2304::Sliced16>;
    if (tokens <= 2432) return launch_bf16_pair_mma<Mma32, LinearIdentityEpilogue>;
    if (tokens <= 5120) return launch_bf16_pair_tma<Tma64, Bf16MmaWarpStoreEpilogue>;
    return launch_bf16_pair_tma<bf16_n32_k2304::Tma128, LinearIdentityEpilogue>;
}

} // namespace

void launch_bf16_pair_n32_k2304(const Tensor& x0, const Tensor& x1, const Weight& w0,
                                const Weight& w1, Tensor& y0, Tensor& y1, cudaStream_t stream) {
    const auto a = bf16_a16_operands(x0, w0), b = bf16_a16_operands(x1, w1);
    const LinearBf16Output first{static_cast<__nv_bfloat16*>(y0.data), a.rows};
    const LinearBf16Output second{static_cast<__nv_bfloat16*>(y1.data), b.rows};
    select(a.tokens)(a, b, first, second, stream);
}

} // namespace ninfer::ops::detail
