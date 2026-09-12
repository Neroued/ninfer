#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/bf16/bf16_n256_k5120.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch_chunk(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                  std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = 256 / Schedule::kOutputRowsPerCta;
    const dim3 grid(kRowTiles, (tokens + Schedule::kTileTokens - 1) / Schedule::kTileTokens);
    static const cudaError_t attr =
        cudaFuncSetAttribute(bf16_n256_k5120_mma_kernel<Schedule>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
    CUDA_CHECK(attr);
    bf16_n256_k5120_mma_kernel<Schedule>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(x, weight, out, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_grid(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kTileTokens, [&](int offset, int count) {
        launch_chunk<Schedule>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(offset) * 5120,
            static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<__nv_bfloat16*>(out.data) + static_cast<std::int64_t>(offset) * 256, count,
            stream);
    });
}
} // namespace

void launch_bf16_n256_k5120(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    // A single grid owns all columns. Wider tiles trade some K parallelism for less duplicated
    // weight staging. At 77 columns cold latency converges and repeated calls favor wider tiles.
    if (x.ne[1] <= 76) {
        launch_grid<Bf16N256K5120MmaSchedule<16, 8>>(x, weight, out, stream);
    } else {
        launch_grid<Bf16N256K5120MmaSchedule<8, 16>>(x, weight, out, stream);
    }
}

} // namespace ninfer::ops::detail
