#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                          = W8DFlash2AttentionProjectionGeometry;
constexpr std::int32_t kQueryRows       = 4096;
constexpr std::int32_t kKvRows          = 1024;
constexpr std::int32_t kLastSmallTokens = 48;
using Output                            = W8SplitOutput3<kQueryRows, kKvRows, kKvRows>;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

// Exact input-load unrolling matters in the first two MMA column tiles. Wider blocks use
// runtime live columns with one specialization per accumulator capacity.
template <int Columns, bool Exact>
void launch_small(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    constexpr int Capacity = (Columns + 7) / 8 * 8;
    constexpr int Warps = Columns <= 4 ? 16 : Columns <= 16 ? 8 : 4;
    constexpr auto Scales = Columns <= 4 || (Columns > 8 && Columns <= 16)
                                ? W8SmallTMmaScaleAccess::Direct
                                : W8SmallTMmaScaleAccess::Shared;
    using Schedule = W8SmallTMmaSchedule<Warps, Capacity, 2, Scales>;
    static_assert((kQueryRows % Schedule::kRowsPerCta) == 0);
    static_assert((kKvRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, Columns, Schedule, Output, W8SmallTMmaStoreEpilogue,
                          W8SmallTMmaIdentityRows, false, !Exact>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaStoreEpilogue{},
            W8SmallTMmaIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Exact, std::size_t... I>
constexpr auto make_small_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &launch_small<Exact ? 1 + static_cast<int>(I) : 24 + 8 * static_cast<int>(I), Exact>...};
}
constexpr auto kExactLaunchers = make_small_launchers<true>(std::make_index_sequence<16>{});
constexpr auto kTileLaunchers = make_small_launchers<false>(std::make_index_sequence<4>{});

template <class Schedule, bool Full>
void launch_mma_slice(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                      cudaStream_t stream) {
    static_assert((kQueryRows % Schedule::BM) == 0);
    static_assert((kKvRows % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(Geometry::kOutputRows / Schedule::BM,
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Geometry::kOutputRows,
            Geometry::kInputRows, x.ne[1], Geometry::kInputRows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool AllowFull = true>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        if constexpr (AllowFull) {
            if ((count % Schedule::BN) == 0) {
                launch_mma_slice<Schedule, true>(x_slice, weight, q_slice, k_slice, v_slice,
                                                 stream);
                return;
            }
        }
        launch_mma_slice<Schedule, false>(x_slice, weight, q_slice, k_slice, v_slice, stream);
    });
}

} // namespace

void w8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kW8DFlash2AttentionFirstSmallT || x.ne[1] > kLastSmallTokens) {
        throw std::invalid_argument("W8 DFlash2 attention input small-T: unsupported T");
    }
    if (x.ne[1] <= 16)
        kExactLaunchers[x.ne[1] - 1](x, weight, q, k, v, stream);
    else
        kTileLaunchers[(x.ne[1] - 17) / 8](x, weight, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<16, 64, 16, 16, 1, 2, 128, 1>;
    // This route owns only the partial 49..63-column tile.
    launch_mma<Schedule, false>(x, w, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 16, 16, 1, 2, 128, 1>;
    launch_mma<Schedule>(x, w, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    // Three-block launch bounds reduce register usage and keep all 384 decode CTAs in one wave.
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 16, 16, 3, 2, 128, 1>;
    launch_mma<Schedule>(x, w, q, k, v, stream);
}

} // namespace ninfer::ops::detail
