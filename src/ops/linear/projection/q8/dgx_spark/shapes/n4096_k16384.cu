#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma_prefetch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <int Tokens, int Warps>
using Small =
    Q8A16SlicedKMmaSchedule<Tokens, Warps, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                            Q8ActivationStage::PaddedZero, 16384, Tokens, false, 1>;
using Hot64    = Q8A16TmaPrefetchSchedule<32, 64, 32, 16, 4, 16384>;
using Medium96 = Q8A16TmaPrefetchSchedule<64, 96, 64, 24, 1, 16384, 2>;
using Wide128  = Q8A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 16384>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, weight),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void prefetched(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_prefetch<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}

void bulk(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_k32<Q8A16TmaK32PrefetchSchedule<8, 16384, true>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}
} // namespace

Q8Launch select_q8_n4096_k16384(std::int32_t tokens) {
    if (tokens == 1) return gemv<Q8A16GemvSchedule<2, 8, 2, 16384, 4>>;
    if (tokens == 2) return gemv<Q8A16GemvTileSchedule<2, 2, 8, 2, 16384, 4>>;
    if (tokens <= 4) return gemv<Q8A16GemvTileSchedule<4, 2, 8, 2, 16384, 4>>;
    if (tokens <= 8) return sliced<Small<8, 16>>;
    if (tokens <= 16) return sliced<Small<16, 32>>;
    if (tokens <= 32) return sliced<Small<32, 8>>;
    if (tokens <= 40) return sliced<Small<40, 4>>;
    if (tokens <= 48) return sliced<Small<48, 4>>;
    if (tokens <= 64) return prefetched<Hot64>;
    if (tokens <= 128) return prefetched<Wide128>;
    if (tokens <= 288) return prefetched<Medium96>;
    if (tokens <= 1024) return prefetched<Wide128>;
    return bulk;
}
} // namespace ninfer::ops::detail
