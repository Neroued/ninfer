#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
using TargetOutput           = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = W8SplitOutput3<4096, 1024, 1024>;

template <class Schedule, W8KernelVariant Variant, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Variant, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, Rows, kHidden, x.ne[1], kHidden);
}

template <class Schedule, int Rows, class Output>
void dispatch_variant(W8KernelVariant variant, const Tensor& x, const Weight& weight, Output output,
                      cudaStream_t stream) {
    if (variant == W8KernelVariant::Full) {
        launch_variant<Schedule, W8KernelVariant::Full, Rows>(x, weight, output, stream);
    } else if (variant == W8KernelVariant::Predicated) {
        launch_variant<Schedule, W8KernelVariant::Predicated, Rows>(x, weight, output, stream);
    } else {
        throw std::invalid_argument("W8 attention input MMA requires a tiled variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_attn_input_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kTargetRows>(variant, x, weight, output, stream);
}

void w8_attn_input_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kTargetRows>(variant, x, weight, output, stream);
}

void w8_attn_input_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                       cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_attn_input_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                       cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                 const Weight& weight, Tensor& q, Tensor& k,
                                                 Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

void w8_companion_attn_input_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x,
                                                 const Weight& weight, Tensor& q, Tensor& k,
                                                 Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    dispatch_variant<Schedule, kCompanionRows>(variant, x, weight, output, stream);
}

} // namespace ninfer::ops::detail
