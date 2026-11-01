#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void launch_q8_n4096_k1024_bulk(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream) {
    launch_q8_a16_tma_k32<Q8A16TmaK32Schedule<4, 1024, true>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<true>{}, stream);
}
} // namespace ninfer::ops::detail
