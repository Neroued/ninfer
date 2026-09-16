#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           Nvfp4ScaleLayout layout, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    const int blocks          = (tasks + kThreads - 1) / kThreads;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    // The tiled plane exists only for a K whose group count divides into whole tiles. Registered
    // Ks all do; gating the instantiation keeps a future K that does not out of a hard compile
    // error and into a runtime message.
    if constexpr (ActivationGeometry::kGroupsPerRow % kNvfp4ScaleTileGroups == 0) {
        if (layout == Nvfp4ScaleLayout::Tiled) {
            nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled>
                <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales, tokens,
                                                  weight.input_scale_divisor);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else if (layout == Nvfp4ScaleLayout::Tiled) {
        throw std::invalid_argument("nvfp4 W4A4 tiled scales need K groups in whole tiles");
    }
    nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::RowMajor>
        <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales, tokens,
                                          weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                Nvfp4ScaleLayout layout, cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    // The tiled layout is a bijection onto the scale plane only for a whole number of token tiles.
    // Check it where the layout is acted on rather than trusting each caller's own predicate.
    if (layout == Nvfp4ScaleLayout::Tiled && (x.ne[1] % kNvfp4TmaBlockM) != 0) {
        throw std::invalid_argument("nvfp4 W4A4 tiled scales need a whole number of token tiles");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, layout, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail
