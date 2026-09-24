#include "core/weight.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q6/q6_rowsplit_gemm_mma.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using MmaR64C16K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 16, 128, 16, 8, 2, 3, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C24K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 24, 128, 16, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C32K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 32, 128, 16, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C40K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 40, 128, 32, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C48K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 48, 128, 16, 16, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C56K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 56, 128, 32, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR32C16K256Schedule =
    Q6RowSplitMmaGemmSchedule<32, 16, 256, 16, 8, 2, 3, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR32C24K256Schedule =
    Q6RowSplitMmaGemmSchedule<32, 24, 256, 16, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR32C32K256Schedule =
    Q6RowSplitMmaGemmSchedule<32, 32, 256, 16, 8, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C64Schedule =
    Q6RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q6FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q6ScaleLoad::Scalar16>;
using MmaR64C64K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 64, 128, 16, 16, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C72K128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 72, 128, 32, 24, 2, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C80Schedule =
    Q6RowSplitMmaGemmSchedule<64, 80, 128, 16, 40, 1, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C96Schedule =
    Q6RowSplitMmaGemmSchedule<64, 96, 128, 16, 48, 1, 2, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64C112PartialSchedule =
    Q6RowSplitMmaGemmSchedule<64, 112, 64, 64, 16, 2, 1, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32>;
using MmaR64C112Schedule =
    Q6RowSplitMmaGemmSchedule<64, 112, 64, 16, 112, 2, 1, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32>;
using MmaR64C128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32>;

template <class Cfg, bool Full>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, Cfg::kBlockRows)),
                    static_cast<unsigned>(div_up(t, Cfg::kBlockCols)), 1u);

    q6_rowsplit_gemm_mma_kernel<Cfg, Full>
        <<<grid, Cfg::kThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (out.ne[0] % Schedule::kBlockRows) == 0 &&
                      (x.ne[1] % Schedule::kBlockCols) == 0 && x.ne[0] == w.padded_shape[1] &&
                      (x.ne[0] % Schedule::kBlockK) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             if (full) {
                                 launch_schedule<Schedule, true>(x_slice, w, out_slice, stream);
                             } else {
                                 launch_schedule<Schedule, false>(x_slice, w, out_slice, stream);
                             }
                         });
}

} // namespace

void launch_q6_mma_r64_c16_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C16K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c24_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C24K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c32_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C32K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c40_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C40K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c48_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C48K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c56_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C56K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r32_c16_k256(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR32C16K256Schedule>(x, w, out, stream);
}

void launch_q6_mma_r32_c24_k256(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR32C24K256Schedule>(x, w, out, stream);
}

void launch_q6_mma_r32_c32_k256(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR32C32K256Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C64Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c64_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C64K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c72_k128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_route<MmaR64C72K128Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C80Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C96Schedule>(x, w, out, stream);
}

void launch_q6_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // A complete column tile uses a wider fragment mapping; tails retain the compact mapping.
    if (x.ne[1] % 112 == 0) {
        launch_route<MmaR64C112Schedule>(x, w, out, stream);
    } else {
        launch_route<MmaR64C112PartialSchedule>(x, w, out, stream);
    }
}

void launch_q6_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C128Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
