#include "core/weight.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C64Schedule =
    Q5RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q5FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q5ScaleLoad::Scalar16>;
using MmaR64C128Schedule =
    Q5RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
// Narrow-column tiles: a 128-wide tile spends a whole wave per four column tiles regardless of how
// many of its columns are live, so the row geometries whose T stays narrow are cheaper here.
using MmaR64C16Schedule =
    Q5RowSplitMmaGemmSchedule<64, 16, 64, 16, 8, 2, 3, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C32S3Schedule =
    Q5RowSplitMmaGemmSchedule<64, 32, 64, 16, 16, 3, 2, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR32C128Schedule =
    Q5RowSplitMmaGemmSchedule<32, 128, 64, 32, 32, 2, 1, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;

template <class Schedule, bool Full, Q5MmaEpilogue Epilogue>
void launch_kernel(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    const auto* residual        = Epilogue == Q5MmaEpilogue::AddResidual
                                      ? static_cast<const __nv_bfloat16*>(out.data)
                                      : nullptr;
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    q5_rowsplit_gemm_mma_kernel<Schedule, Full, Epilogue><<<grid, Schedule::kThreads, 0, stream>>>(
        xp, codes, high, scales, residual, outp, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (w.n % 64) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % 64) == 0;
    for_each_token_slice(
        x.ne[1], Schedule::kBlockCols, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            if (full) {
                launch_kernel<Schedule, true, Q5MmaEpilogue::Store>(x_slice, w, out_slice, stream);
            } else {
                launch_kernel<Schedule, false, Q5MmaEpilogue::Store>(x_slice, w, out_slice, stream);
            }
        });
}

} // namespace

void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C64Schedule>(x, w, out, stream);
}

void launch_q5_mma_r64_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C16Schedule>(x, w, out, stream);
}

void launch_q5_mma_r64_c32_s3(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C32S3Schedule>(x, w, out, stream);
}

void launch_q5_mma_r32_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR32C128Schedule>(x, w, out, stream);
}

void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<MmaR64C128Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
