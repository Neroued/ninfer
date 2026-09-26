#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_simt_launch.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;

constexpr int kRowsPerCta = 8;
constexpr int kStages     = 2;
constexpr int kCols       = 4;
using TargetOutput        = LinearBf16SegmentedOutput<4096, 512, 4096, 512>;
using CompanionOutput     = LinearBf16SegmentedOutput<4096, 1024, 1024>;

template <int Rows, class Output>
void launch_route(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    using Schedule = Q8A16SimtSchedule<kRowsPerCta, kCols, 1, 32, kStages, Cache::cg, 1>;
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                 stream);
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
