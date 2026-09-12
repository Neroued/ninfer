#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool Full>
void launch_slice(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), rows};
    w8_rowsplit_gemm_mma_kernel<Schedule, Full><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (w.n % Schedule::BM) == 0 && (x.ne[1] % Schedule::BN) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % Schedule::BK) == 0;
    for_each_token_slice(x.ne[1], Schedule::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_slice<Schedule, true>(x_slice, w, out_slice, stream);
        } else {
            launch_slice<Schedule, false>(x_slice, w, out_slice, stream);
        }
    });
}

template <std::int32_t TileCols>
void launch_exact_tail(W8Launch prefix_launch, const Tensor& x, const Weight& w, Tensor& out,
                       cudaStream_t stream) {
    const std::int32_t full_cols = (x.ne[1] / TileCols) * TileCols;
    if (full_cols <= 0) {
        throw std::invalid_argument("w8 exact-tail route requires a non-empty MMA prefix");
    }

    const Tensor x_prefix = x.slice(1, 0, full_cols);
    Tensor out_prefix     = out.slice(1, 0, full_cols);
    prefix_launch(x_prefix, w, out_prefix, stream);

    const std::int32_t tail = x.ne[1] - full_cols;
    if (tail < 1 || tail > 65) {
        throw std::invalid_argument("w8 exact-tail route requires tail=1..65");
    }
    const Tensor x_tail = x.slice(1, full_cols, tail);
    Tensor out_tail     = out.slice(1, full_cols, tail);
    if (tail == 1) {
        launch_w8_decode_r4(x_tail, w, out_tail, stream);
    } else if (tail <= 32) {
        launch_w8_exact_t_splitk(x_tail, w, out_tail, stream);
    } else {
        launch_w8_exact_t_composite(x_tail, w, out_tail, stream);
    }
}

using MmaR32C64  = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
using MmaR32C96  = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
using MmaR32C128 = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
using MmaR48C64  = W8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>;
using MmaR48C96  = W8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>;
using MmaR48C112 = W8RowSplitMmaGemmSchedule<48, 112, 48, 16, 2>;
using MmaR48C128 = W8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2>;
using MmaR64C96  = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
using MmaR64C112 = W8RowSplitMmaGemmSchedule<64, 112, 64, 16, 2>;
using MmaR64C128 = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
using MmaR96C96  = W8RowSplitMmaGemmSchedule<96, 96, 48, 16, 2>;
using MmaR128C64 = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
using MmaR128C80 = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
// Full-vocabulary cold-cache winners on either side of the 48-column frontier.
using MmaR64x16C48K128A1 = W8RowSplitMmaGemmSchedule<64, 48, 16, 24, 2, 2, 128, 1>;
using MmaR64x32C64K128A1 = W8RowSplitMmaGemmSchedule<64, 64, 32, 16, 2, 2, 128, 1>;

} // namespace

#define NINFER_W8_MMA_LAUNCHER(Name, Schedule)                                                     \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        launch_route<Schedule>(x, w, out, stream);                                                 \
    }

NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r32_c64, MmaR32C64)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r32_c96, MmaR32C96)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r32_c128, MmaR32C128)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r48_c64, MmaR48C64)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r48_c96, MmaR48C96)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r48_c112, MmaR48C112)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r48_c128, MmaR48C128)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64_c96, MmaR64C96)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64_c112, MmaR64C112)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64_c128, MmaR64C128)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r96_c96, MmaR96C96)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r128_c64, MmaR128C64)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r128_c80, MmaR128C80)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64x16_c48_k128_a1, MmaR64x16C48K128A1)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64x32_c64_k128_a1, MmaR64x32C64K128A1)

#undef NINFER_W8_MMA_LAUNCHER

#define NINFER_W8_EXACT_LAUNCHER(Name, Prefix, TileCols)                                           \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        launch_exact_tail<TileCols>(Prefix, x, w, out, stream);                                    \
    }

NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r32_c96, launch_w8_mma_r32_c96, 96)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r32_c128, launch_w8_mma_r32_c128, 128)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r48_c96, launch_w8_mma_r48_c96, 96)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r48_c128, launch_w8_mma_r48_c128, 128)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r64_c96, launch_w8_mma_r64_c96, 96)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r64_c128, launch_w8_mma_r64_c128, 128)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r96_c96, launch_w8_mma_r96_c96, 96)
NINFER_W8_EXACT_LAUNCHER(launch_w8_exact_mma_r128_c80, launch_w8_mma_r128_c80, 80)

#undef NINFER_W8_EXACT_LAUNCHER

} // namespace ninfer::ops::detail
