#pragma once
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include <algorithm>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
using Nvfp4A4Launch = void (*)(const Weight&, Tensor&, Nvfp4W4a4Workspace, std::int32_t,
                               cudaStream_t);

// A selected A4 route: the GEMM to run, and the activation-scale layout that GEMM reads. They
// travel together because the quantizer writes the plane before the GEMM runs and the two must
// agree; a shape selects a route rather than selecting the two halves separately.
struct Nvfp4A4Route {
    Nvfp4A4Launch launch;
    Nvfp4ScaleLayout scales;
};

template <class Geometry, class Schedule>
void launch_nvfp4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    nvfp4_gemv_kernel<Geometry, Schedule>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), 1.0f / weight.weight_scale_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Capacity, class Schedule, bool FullColumns = false>
void launch_nvfp4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int blocks = Geometry::kOutputRows / Schedule::kRowsPerCta *
                           ((Capacity + Schedule::kTokenTile - 1) / Schedule::kTokenTile);
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    nvfp4_simt_kernel<Geometry, Capacity, Schedule, Nvfp4IdentityEpilogue, Nvfp4ContiguousOutput,
                      Nvfp4SimtFinalization::Elementwise, !FullColumns>
        <<<blocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                    static_cast<const std::uint8_t*>(weight.qdata),
                                                    static_cast<const std::uint8_t*>(weight.scales),
                                                    1.0f / weight.weight_scale_divisor, {}, output,
                                                    x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, template <int> class Schedule, std::size_t... Indices>
constexpr auto nvfp4_exact_launchers(std::index_sequence<Indices...>) {
    return std::array<Nvfp4Launch, sizeof...(Indices)>{
        &launch_nvfp4_simt<Geometry, First + static_cast<int>(Indices),
                           Schedule<First + static_cast<int>(Indices)>, true>...};
}

// Each shape supplies its measured exact interval and schedule.
template <class Geometry, int First, int Last, template <int> class Schedule>
Nvfp4Launch select_nvfp4_exact(std::int32_t tokens) {
    static constexpr auto launchers = nvfp4_exact_launchers<Geometry, First, Schedule>(
        std::make_index_sequence<Last - First + 1>{});
    return launchers.at(static_cast<std::size_t>(tokens - First));
}

template <int Chunk, Nvfp4Launch (*Select)(std::int32_t)>
void launch_nvfp4_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        Select(count)(input, weight, output, stream);
    }
}

template <class Geometry, class Schedule>
void launch_nvfp4_a4_mma(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                         std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <Nvfp4GeometryId Geometry>
void launch_nvfp4_a4_tma(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace scratch,
                         std::int32_t tokens, cudaStream_t stream) {
    const float alpha = 1.0f / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_w4a4_tma_linear(Geometry, scratch.codes, scratch.scales,
                                 static_cast<const std::uint8_t*>(weight.qdata),
                                 static_cast<const std::uint8_t*>(weight.scales),
                                 static_cast<__nv_bfloat16*>(out.data), tokens, alpha, stream);
}

// The TMA GEMM reads the tiled plane; every MMA GEMM reads the row-major one. Stating that here,
// once per kind, is what keeps a shape from pairing them the other way round.
template <Nvfp4GeometryId Geometry>
constexpr Nvfp4A4Route nvfp4_a4_tma_route() {
    return {launch_nvfp4_a4_tma<Geometry>, Nvfp4ScaleLayout::Tiled};
}

template <class Geometry, class Schedule>
constexpr Nvfp4A4Route nvfp4_a4_mma_route() {
    return {launch_nvfp4_a4_mma<Geometry, Schedule>, Nvfp4ScaleLayout::RowMajor};
}

template <Nvfp4A4Route (*Select)(std::int32_t)>
void launch_nvfp4_a4(const Tensor& x, const Weight& weight, Tensor& out, Nvfp4W4a4Workspace scratch,
                     cudaStream_t stream) {
    const Nvfp4A4Route route = Select(x.ne[1]);
    launch_nvfp4_w4a4_quantize(x, weight, scratch, route.scales, stream);
    route.launch(weight, out, scratch, x.ne[1], stream);
}
} // namespace ninfer::ops::detail
