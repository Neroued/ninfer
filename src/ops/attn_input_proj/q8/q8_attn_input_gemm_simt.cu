#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_rowsplit_gemm_simt.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
constexpr int kRowsPerCta    = 8;
constexpr int kStages        = 2;
constexpr int kCols          = 4;
using TargetOutput           = Q8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = Q8SplitOutput3<4096, 1024, 1024>;

template <bool Full, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / kRowsPerCta, static_cast<unsigned>(div_up(x.ne[1], kCols)), 1u);
    q8_rowsplit_gemm_simt_kernel<Q8RowSplitSimtSchedule, kCols, kRowsPerCta, kStages, Full,
                                 Q8Epilogue::Store, Output><<<grid, kRowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Rows, kHidden, x.ne[1], kHidden,
        kHidden / 1024);
}

template <int Rows, class Output>
void launch_route(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    if ((x.ne[1] % kCols) == 0) {
        launch_variant<true, Rows>(x, weight, output, stream);
    } else {
        launch_variant<false, Rows>(x, weight, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<kTargetRows>(x, weight, output, stream);
}

void q8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<kCompanionRows>(x, weight, output, stream);
}

} // namespace ninfer::ops::detail
