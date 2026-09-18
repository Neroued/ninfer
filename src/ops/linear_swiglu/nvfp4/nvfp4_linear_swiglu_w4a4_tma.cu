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
// The descriptor below divides the padded token count by this tile to get the plane's tile count,
// and nvfp4_w4a4_padded_tokens rounds up to kNvfp4TmaBlockM, so the two have to be the same number
// or that division stops being exact and the last tile addresses past the plane.
static_assert(M256N128S3::kBlockM == kNvfp4TmaBlockM,
              "the fused descriptor below addresses the quantizer's scale tiles");

template <class Geometry, class Schedule>
Nvfp4W4a4TmaDescriptors make_descriptors(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, std::int32_t tokens) {
    constexpr std::uint32_t kCodeColumns = 64;
    // Activation scales arrive tile-contiguous, as for the shared W4A4 route; see
    // nvfp4_geometry.h for the tile shape the quantizer writes.
    constexpr std::uint32_t kScaleTileGroups = kNvfp4ScaleTileGroups;
    constexpr std::uint64_t kScaleTilesPerPlane =
        static_cast<std::uint64_t>(Geometry::kGroupsPerRow) / kScaleTileGroups;
    constexpr std::uint32_t kPairN = Schedule::kBlockN / 2;
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
    // dim1 counts whole token tiles, so the plane is described over the padded token count the
    // quantizer writes. Describing it over the real count instead would leave the last tile's
    // scales outside the descriptor, where TMA zero-fills them, and every token in that tile would
    // be written out as zero.
    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Schedule::kBlockM,
        (static_cast<std::uint64_t>(nvfp4_w4a4_padded_tokens(tokens)) / Schedule::kBlockM) *
            kScaleTilesPerPlane * kScaleTileGroups,
        Schedule::kBlockM, Schedule::kBlockM, kScaleTileGroups, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode LinearSwiGLU activation scales TMA");
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
    if (tokens <= 0) {
        throw std::invalid_argument("nvfp4 LinearSwiGLU TMA needs a positive token count");
    }

    using Geometry                     = Nvfp4N34816K5120;
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
    // The last M tile may be partial; the kernel bounds its stores by the real token count.
    const dim3 grid((Geometry::kOutputRows / 2) / kPairN,
                    (tokens + M256N128S3::kBlockM - 1) / M256N128S3::kBlockM);
    nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
        <<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(descriptors, alpha, output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
