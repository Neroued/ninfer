#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
using TargetOutput           = Q8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = Q8SplitOutput3<4096, 1024, 1024>;

template <class Schedule, bool Full, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, Rows, kHidden, x.ne[1], kHidden);
}

template <class Schedule, int Rows, class Output>
void launch_route(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<Schedule, true, Rows>(x, weight, output, stream);
    } else {
        launch_variant<Schedule, false, Rows>(x, weight, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r64_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r32_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r64_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r128_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r128_c80_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

} // namespace ninfer::ops::detail
