#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;

using TargetOutput    = LinearBf16SegmentedOutput<4096, 512, 4096, 512>;
using CompanionOutput = LinearBf16SegmentedOutput<4096, 1024, 1024>;

template <class Schedule, int Rows, class Output>
void launch_route(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}

} // namespace

void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (512 % Schedule::kBlockRows) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (512 % Schedule::kBlockRows) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r64_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r32_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r64_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r128_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

void q8_companion_attn_input_mma_r128_c80_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::kBlockRows) == 0 && (1024 % Schedule::kBlockRows) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows>(x, weight, output, stream);
}

} // namespace ninfer::ops::detail
