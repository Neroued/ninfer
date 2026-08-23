#include "ops/linear/fp8/fp8_a8_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry, int Threads = 256>
__global__ __launch_bounds__(Threads,
                             2) void fp8_a8_quantize_kernel(const __nv_bfloat16* __restrict__ input,
                                                            std::uint8_t* __restrict__ codes,
                                                            float* __restrict__ scales) {
    static_assert((ActivationGeometry::kInputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_token  = ActivationGeometry::kInputRows / 2;
    constexpr int pairs_per_thread = pairs_per_token / Threads;
    constexpr int warps            = Threads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float token_scale;

    const int token         = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    const int lane          = tid & 31;
    const int warp          = tid >> 5;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(
        input + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    auto* output_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        values[item]   = bf16x2_bits_to_float2(input_pairs[pair]);
        maximum        = fmaxf(maximum, fabsf(values[item].x));
        maximum        = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = warp_max(maximum);
        if (lane == 0) { token_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = token_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair      = tid + item * Threads;
        const float2 scaled = make_float2(values[item].x * inverse, values[item].y * inverse);
        output_pairs[pair]  = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scales[token] = scale; }
}

template <class Geometry, class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    const int row_tiles   = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks      = row_tiles * token_tiles;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue,
                           Fp8ContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, Fp8A8Workspace workspace, cudaStream_t stream) {
    constexpr int kThreads = 256;
    fp8_a8_quantize_kernel<ActivationGeometry, kThreads><<<x.ne[1], kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Schedule, true>(weight, out, workspace, tokens, stream);
    } else {
        launch_mma<Geometry, Schedule, false>(weight, out, workspace, tokens, stream);
    }
}

} // namespace

void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 A8 requires caller workspace");
    }
    switch (weight.k) {
    case Fp8Activation5120Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation5120Geometry>(x, workspace, stream);
        return;
    case Fp8Activation6144Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation6144Geometry>(x, workspace, stream);
        return;
    case Fp8Activation17408Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation17408Geometry>(x, workspace, stream);
        return;
    default:
        throw std::invalid_argument("fp8 A8 quantize: unsupported K");
    }
}

void launch_fp8_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace workspace,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::AttnInput:
        launch_problem<Fp8AttnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Fp8Problem::GdnInput:
        launch_problem<Fp8GdnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Fp8Problem::MlpGateUp:
        launch_problem<Fp8MlpGateUpGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Fp8Problem::Vocabulary:
        break;
    case Fp8Problem::Residual6144:
        launch_problem<Fp8Residual6144Geometry>(weight, out, workspace, tokens, stream);
        return;
    case Fp8Problem::Residual17408:
        launch_problem<Fp8Residual17408Geometry>(weight, out, workspace, tokens, stream);
        return;
    }
    throw std::logic_error("FP8 vocabulary has no A8 route");
}

} // namespace ninfer::ops::detail
