#include "ops/linear/w8/w8_rowsplit_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kHidden         = 16384;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 32;
using ExactTLauncher          = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    static_assert((kRows % kRowsPerCta) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols>
        <<<kRows / kRowsPerCta, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ExactTLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

void require_problem(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.ne[0] != kHidden || out.ne[0] != kRows || out.ne[1] != x.ne[1] || w.n != kRows ||
        w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("W8 exact-T split-K requires [2048,16384]");
    }
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, x.ne[1]);
}

} // namespace

void w8_rowsplit_exact_t_splitk_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 exact-T split-K requires T=2..32");
    }
    kLaunchers[x.ne[1] - kFirstExactCols](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void w8_rowsplit_exact_t_composite_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < 33 || x.ne[1] > 127) {
        throw std::invalid_argument("W8 exact-T composite requires T=33..127");
    }

    std::int32_t offset = 0;
    while (x.ne[1] - offset >= 32) {
        const Tensor x_slice = x.slice(1, offset, 32);
        Tensor out_slice     = out.slice(1, offset, 32);
        w8_rowsplit_exact_t_splitk_launch(x_slice, w, out_slice, stream);
        offset += 32;
    }
    const std::int32_t tail = x.ne[1] - offset;
    if (tail == 1) {
        const Tensor x_slice = x.slice(1, offset, 1);
        Tensor out_slice     = out.slice(1, offset, 1);
        w8_rowsplit_decode_r16_launch(x_slice, w, out_slice, stream);
    } else if (tail >= 2) {
        const Tensor x_slice = x.slice(1, offset, tail);
        Tensor out_slice     = out.slice(1, offset, tail);
        w8_rowsplit_exact_t_splitk_launch(x_slice, w, out_slice, stream);
    }
}

void w8_rowsplit_medium_t_splitk_launch(W8ScheduleId schedule, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    switch (schedule) {
    case W8ScheduleId::SplitKMediumC48:
        if (x.ne[1] > 48) { break; }
        launch_medium<48, 4, 2, 3>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    case W8ScheduleId::SplitKMediumC64:
        if (x.ne[1] > 64) { break; }
        launch_medium<64, 4, 2, 2>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    case W8ScheduleId::SplitKMediumC96:
        if (x.ne[1] > 96) { break; }
        launch_medium<96, 2, 6, 3>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    case W8ScheduleId::SplitKMediumC128:
        if (x.ne[1] > 128) { break; }
        launch_medium<128, 2, 4, 2>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    case W8ScheduleId::SplitKMediumC144:
        if (x.ne[1] > 144) { break; }
        launch_medium<144, 2, 9, 2>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    case W8ScheduleId::SplitKMediumC160:
        if (x.ne[1] > 160) { break; }
        launch_medium<160, 2, 5, 2>(x, w, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    default:
        break;
    }
    throw std::invalid_argument("W8 medium-T split-K schedule does not cover this T");
}

} // namespace ninfer::ops::detail
