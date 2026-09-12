#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using M256N128S3 = Nvfp4W4a4TmaSchedule<256, 3, 1>;

template <class Geometry, class Schedule>
Nvfp4W4a4TmaDescriptors make_descriptors(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, std::int32_t tokens) {
    constexpr std::uint32_t kCodeColumns  = 64;
    constexpr std::uint32_t kScaleColumns = 16;
    constexpr std::uint32_t kPairN        = Schedule::kBlockN / 2;
    constexpr std::uint64_t kWeightScaleBytes =
        static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;

    Nvfp4W4a4TmaDescriptors descriptors{};
    descriptors.a_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, tokens, Geometry::kCodeBytesPerRow, kCodeColumns,
        Schedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_64B, "encode LinearSwiGLU activation codes TMA");
    descriptors.b_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow, kCodeColumns,
        kPairN, CU_TENSOR_MAP_SWIZZLE_64B, "encode LinearSwiGLU weight codes TMA");
    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns, Schedule::kBlockM,
        CU_TENSOR_MAP_SWIZZLE_NONE, "encode LinearSwiGLU activation scales TMA");
    descriptors.b_scales =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          16, kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                          "encode LinearSwiGLU weight scales TMA");
    return descriptors;
}

} // namespace

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream) {
    if (tokens < M256N128S3::kBlockM || (tokens % M256N128S3::kBlockM) != 0) {
        throw std::invalid_argument(
            "nvfp4 LinearSwiGLU TMA requires a positive M256 full-tile token count");
    }

    using Geometry                     = Nvfp4MlpGateUpGeometry;
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4LinearSwiGluTmaSharedStorage<M256N128S3>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const Nvfp4W4a4TmaDescriptors descriptors = make_descriptors<Geometry, M256N128S3>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr int kPairN = M256N128S3::kBlockN / 2;
    const dim3 grid((Geometry::kOutputRows / 2) / kPairN, tokens / M256N128S3::kBlockM);

    // MSVC rejects over-aligned kernel parameters (C2711), so the descriptors are
    // passed to the kernel by device pointer instead of by value on Windows only.
    // cudaMallocAsync keeps the entire alloc-copy-launch-free cycle stream-ordered;
    // the stream memory pool reuses the same slot on subsequent launches.
#if defined(_MSC_VER)
    Nvfp4W4a4TmaDescriptors* device_descriptors = nullptr;
    CUDA_CHECK(cudaMallocAsync(&device_descriptors, sizeof(Nvfp4W4a4TmaDescriptors), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_descriptors, &descriptors, sizeof(Nvfp4W4a4TmaDescriptors),
                               cudaMemcpyHostToDevice, stream));
#endif
#if defined(_MSC_VER)
    nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
        <<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(device_descriptors, alpha, output);
#else
    nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
        <<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(descriptors, alpha, output);
#endif
    CUDA_CHECK(cudaGetLastError());
#if defined(_MSC_VER)
    CUDA_CHECK(cudaFreeAsync(device_descriptors, stream));
#endif
}

} // namespace ninfer::ops::detail
