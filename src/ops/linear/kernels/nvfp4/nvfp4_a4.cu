#include "core/weight.h"
#include "core/platform.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

#include "core/device.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_quantize.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                           Nvfp4ScaleLayout layout, cudaStream_t stream, bool programmatic) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    // The tiled plane is allocated and addressed over whole tiles, so its launch has to reach the
    // padding of the last one; the row-major plane stops at the real token count.
    const std::int32_t written_tokens =
        layout != Nvfp4ScaleLayout::RowMajor ? nvfp4_a4_padded_tokens(tokens, layout) : tokens;
    const std::int32_t tasks = written_tokens * ActivationGeometry::kGroupsPerRow;
    const int blocks         = (tasks + kThreads - 1) / kThreads;
    const auto launch        = [&]<Nvfp4ScaleLayout Layout>() {
        const auto submit = [&]<bool Programmatic>() {
            nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Layout, __nv_bfloat16,
                                            Programmatic>
                <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales, tokens,
                                                         written_tokens, weight.input_scale);
        };
        if (programmatic)
            submit.template operator()<true>();
        else
            submit.template operator()<false>();
    };
    if (layout == Nvfp4ScaleLayout::RowMajor) {
        launch.template operator()<Nvfp4ScaleLayout::RowMajor>();
    } else if constexpr (ActivationGeometry::kGroupsPerRow % kNvfp4ScaleTileGroups == 0) {
        if (layout == Nvfp4ScaleLayout::Tiled128)
            launch.template operator()<Nvfp4ScaleLayout::Tiled128>();
        else
            launch.template operator()<Nvfp4ScaleLayout::Tiled256>();
    } else {
        throw std::invalid_argument("nvfp4 A4 tiled scales need K groups in whole tiles");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream, bool programmatic) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 A4 requires caller workspace");
    }
    // The tiled layout writes the padding of the last tile, so it needs a plane allocated over the
    // padded token count. Check that here, where the layout is acted on, rather than trust each
    // caller to have sized it that way.
    if (layout != Nvfp4ScaleLayout::RowMajor) {
        const std::size_t needed =
            static_cast<std::size_t>(nvfp4_a4_padded_tokens(x.ne[1], layout)) *
            (static_cast<std::size_t>(weight.k) / 16);
        if (workspace.scale_bytes < needed) {
            throw std::invalid_argument(
                "nvfp4 A4 tiled scales need a plane allocated over whole token tiles");
        }
    }
    switch (weight.k) {
#if NINFER_TARGET_SPARK
    case 4096:
        launch_quantize_exact<Nvfp4ActivationGeometry<4096>>(x, weight, workspace, layout, stream,
                                                             programmatic);
        return;
#endif
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, layout, stream,
                                                           programmatic);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, layout, stream,
                                                           programmatic);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, layout, stream,
                                                            programmatic);
        return;
#if NINFER_TARGET_RTX5090
    case Nvfp4Activation14336Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation14336Geometry>(x, weight, workspace, layout, stream,
                                                            programmatic);
        return;
#endif
    default:
        throw std::invalid_argument("nvfp4 A4 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail
