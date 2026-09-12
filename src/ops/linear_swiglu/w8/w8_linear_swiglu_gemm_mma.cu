#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool Full>
void launch_variant(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    const dim3 grid(out.ne[0] / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(w.qdata),
                                                 static_cast<const std::uint8_t*>(w.scales), output,
                                                 w.n, w.k, x.ne[1], w.padded_shape[1]);
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<Schedule, true>(x, w, out, stream);
    } else {
        launch_variant<Schedule, false>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 16, 16, 3, 2, 128, 1>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 32, 16, 2, 2, 128, 1>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 8, 2, 2, 128, 1>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 8, 2, 2, 128, 1>;
    launch_route<Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
