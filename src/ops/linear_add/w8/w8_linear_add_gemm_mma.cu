#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, W8KernelVariant Variant>
void launch_tt(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), rows};
    w8_rowsplit_gemm_mma_kernel<Schedule, Variant, W8Epilogue::Residual>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
}

template <class Schedule>
void launch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& residual_out,
                    cudaStream_t stream) {
    switch (variant) {
    case W8KernelVariant::Full:
        launch_tt<Schedule, W8KernelVariant::Full>(x, w, residual_out, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_tt<Schedule, W8KernelVariant::Predicated>(x, w, residual_out, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 linear_add MMA requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_add_mma_r32_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 112, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r48_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r48_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 80, 48, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r48_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r48_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 112, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r48_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 112, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

void w8_linear_add_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
