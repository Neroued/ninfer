#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGateUpRows   = 12288;
constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;

template <class Schedule, W8KernelVariant Variant>
void launch_variant(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert((kIntermediate % (Schedule::BM / 2)) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const dim3 grid(kIntermediate / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Variant, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(w.qdata),
                                                 static_cast<const std::uint8_t*>(w.scales), output,
                                                 kGateUpRows, kHidden, x.ne[1], kHidden);
}

template <class Schedule>
void dispatch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                      cudaStream_t stream) {
    if (variant == W8KernelVariant::Full) {
        launch_variant<Schedule, W8KernelVariant::Full>(x, w, out, stream);
    } else if (variant == W8KernelVariant::Predicated) {
        launch_variant<Schedule, W8KernelVariant::Predicated>(x, w, out, stream);
    } else {
        throw std::invalid_argument("W8 LinearSwiGLU MMA requires a tiled variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_swiglu_mma_r32_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    dispatch_variant<Schedule>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
