#pragma once

#include "ops/common/host/launch.h"
#include "ops/linear_attention/gated_delta_net/control/bf16/launch.h"
#include "ops/linear_attention/gated_delta_net/control/kernels/mma.cuh"
#include "core/device.h"
#include <algorithm>
#include <type_traits>

namespace ninfer::ops::detail::gdn_control {

template <class Geometry, int Split, int Warps, bool Normalize = false, int Capacity = 0>
int residency() {
    constexpr int shared      = Geometry::shared_bytes;
    static const int resident = [] {
        int tail = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &tail, mma_kernel<Geometry, Split, false, Warps, Normalize, Capacity>, Warps * 32,
            shared));
        if constexpr (!Normalize) {
            int full = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &full, mma_kernel<Geometry, Split, true, Warps, Normalize, Capacity>, Warps * 32,
                shared));
            return std::min(full, tail);
        } else
            return tail;
    }();
    return resident;
}

template <class Geometry, int Split, int Warps = 8, bool Normalize = false, int Capacity = 0>
bool launch_mma(const Inputs& in, GdnControlOutputs out, Scratch scratch,
                CudaExecutionView execution, const Norm* norm = nullptr) {
    constexpr int tile   = Geometry::kBlockN;
    constexpr int rows   = (Geometry::kPaired ? Geometry::kHeads : 2 * Geometry::kHeads) / kBlockM;
    constexpr int shared = Geometry::shared_bytes;
    const auto submit    = [&](const Tensor& x, GdnControlOutputs result, Tensor normalized) {
        const int columns = x.ne[1];
        const dim3 grid(static_cast<unsigned>((std::int64_t(columns) + tile - 1) / tile), rows,
                           Split);
        const auto invoke = [&]<bool Full>() {
            (void)common::prepare_shared<
                   shared, mma_kernel<Geometry, Split, Full, Warps, Normalize, Capacity>, true>();
            const auto* gain =
                norm ? static_cast<const __nv_bfloat16*>(norm->weight.data) : nullptr;
            const float eps = norm ? norm->eps : 0.0F;
            if constexpr (Split > 1) {
                cudaLaunchAttribute cooperative{};
                cooperative.id              = cudaLaunchAttributeCooperative;
                cooperative.val.cooperative = 1;
                cudaLaunchConfig_t config{};
                config.gridDim          = grid;
                config.blockDim         = dim3(Warps * 32);
                config.dynamicSmemBytes = shared;
                config.stream           = execution.stream();
                config.attrs            = &cooperative;
                config.numAttrs         = 1;
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config, mma_kernel<Geometry, Split, Full, Warps, Normalize, Capacity>,
                    static_cast<const __nv_bfloat16*>(x.data), gain,
                    static_cast<__nv_bfloat16*>(normalized.data), eps,
                    static_cast<const __nv_bfloat16*>(in.a),
                    static_cast<const __nv_bfloat16*>(in.b),
                    static_cast<const float*>(in.a_log.data),
                    static_cast<const float*>(in.dt_bias.data),
                    static_cast<float*>(scratch.partial.data), static_cast<float*>(result.g.data),
                    static_cast<float*>(result.beta.data), columns));
            } else {
                mma_kernel<Geometry, Split, Full, Warps, Normalize, Capacity>
                    <<<grid, Warps * 32, shared, execution.stream()>>>(
                        static_cast<const __nv_bfloat16*>(x.data), gain,
                        static_cast<__nv_bfloat16*>(normalized.data), eps,
                        static_cast<const __nv_bfloat16*>(in.a),
                        static_cast<const __nv_bfloat16*>(in.b),
                        static_cast<const float*>(in.a_log.data),
                        static_cast<const float*>(in.dt_bias.data),
                        static_cast<float*>(scratch.partial.data),
                        static_cast<float*>(result.g.data), static_cast<float*>(result.beta.data),
                        columns);
            }
        };
        if constexpr (Normalize)
            invoke.template operator()<false>();
        else if (columns % tile == 0)
            invoke.template operator()<true>();
        else
            invoke.template operator()<false>();
    };
    if constexpr (Split == 1)
        submit(in.x, out, norm ? norm->normalized : Tensor{});
    else {
        const auto tiles = std::int64_t(execution.multiprocessor_count()) *
                           residency<Geometry, Split, Warps, Normalize, Capacity>() /
                           (rows * Split);
        if (tiles < 1) return false;
        const auto capacity = tiles * tile;
        if (in.x.ne[1] <= capacity)
            submit(in.x, out, norm ? norm->normalized : Tensor{});
        else
            for (int first = 0; first < in.x.ne[1];) {
                const int count =
                    static_cast<int>(std::min<std::int64_t>(in.x.ne[1] - first, capacity));
                submit(in.x.slice(1, first, count),
                       {out.g.slice(1, first, count), out.beta.slice(1, first, count)},
                       norm ? norm->normalized.slice(1, first, count) : Tensor{});
                first += count;
            }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninfer::ops::detail::gdn_control
