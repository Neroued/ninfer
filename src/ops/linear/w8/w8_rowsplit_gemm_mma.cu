#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_kernels.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Cfg, W8KernelVariant Variant>
void launch_w8_cfg(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t m        = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t n        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(m, Cfg::BM)),
                    static_cast<unsigned>(div_up(n, Cfg::BN)), 1u);
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* outp         = static_cast<__nv_bfloat16*>(out.data);
    const W8ContiguousOutput output{outp, m};
    w8_rowsplit_gemm_mma_kernel<Cfg, Variant>
        <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, scales, output, m, k, n, padded_k);
}

template <class Cfg>
void launch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    switch (variant) {
    case W8KernelVariant::Full:
        launch_w8_cfg<Cfg, W8KernelVariant::Full>(x, w, out, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_w8_cfg<Cfg, W8KernelVariant::Predicated>(x, w, out, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 MMA launch requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_rowsplit_gemm_mma_r32_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 112, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r48_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r48_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 80, 48, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r48_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r48_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 112, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r48_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 112, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r96_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<96, 64, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r96_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<96, 80, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r96_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<96, 96, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r96_c112_launch(W8KernelVariant variant, const Tensor& x,
                                          const Weight& w, Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<96, 112, 48, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_variant<Schedule>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
