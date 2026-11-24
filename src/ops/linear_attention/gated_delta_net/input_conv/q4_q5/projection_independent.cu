#include "ops/linear/kernels/q5/q5_instances.cuh"
#include "ops/linear/kernels/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q4_q5/launch.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/kernels/q4/q4_simt_launch.cuh"
#include "ops/linear/kernels/q4/q4_gemv_launch.cuh"
#include "ops/linear/kernels/q5/q5_simt_launch.cuh"
#include "ops/linear/kernels/q5/q5_gemv_launch.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kHidden    = 5120;

using Q4GdnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8T8Schedule = Q4A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kHidden, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

using Q4Launcher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

Q4Launcher select_q4_projection(int columns) {
    switch (columns) {
    case 1:
        return launch_q4_gemv;
    case 2:
    case 3:
    case 4:
        return launch_q4_simt<Q4GdnSimtR8T4Schedule>;
    case 5:
    case 6:
        return launch_q4_simt<Q4GdnSimtR8T8Schedule>;
    case 7:
        return launch_q4_sliced_exact<7>;
    case 8:
        return launch_q4_sliced_exact<8>;
    case 9:
        return launch_q4_sliced_exact<9>;
    case 10:
        return launch_q4_sliced_exact<10>;
    case 11:
        return launch_q4_sliced_exact<11>;
    case 12:
        return launch_q4_sliced_exact<12>;
    }
    throw std::logic_error("GDN independent Q4 projection: invalid selected extent");
}

auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<kValueRows>{
        {static_cast<__nv_bfloat16*>(first.data), std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)),
         0},
        {static_cast<__nv_bfloat16*>(second.data),
         std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)), 0}};
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(q5_linear_operands(x, weight),
                                                           q5_projection_output(value, z),
                                                           LinearIdentityEpilogue{}, stream);
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, kHidden, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight),
                                        q5_projection_output(value, z), LinearIdentityEpilogue{},
                                        stream);
}

template <int kColsPerTile>
void launch_q5_simt_cols(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    using Schedule = Q5A16SimtSchedule<8, kColsPerTile, 1, 16, 2, Cache::ca, 1>;
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(value, z),
                                 LinearIdentityEpilogue{}, stream);
}

using Q5Launcher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

Q5Launcher select_q5_projection(int columns) {
    switch (columns) {
    case 1:
        return launch_q5_gemv;
    case 2:
        return launch_q5_split4<2>;
    case 3:
        return launch_q5_split4<3>;
    case 4:
        return launch_q5_split4<4>;
    case 5:
        return launch_q5_split4<5>;
    case 6:
        return launch_q5_split4<6>;
    case 7:
        return launch_q5_split4<7>;
    case 8:
        return launch_q5_split4<8>;
    case 9:
        return launch_q5_split4<9>;
    case 10:
        return launch_q5_split4<10>;
    case 11:
    case 12:
        return launch_q5_simt_cols<4>;
    }
    throw std::logic_error("GDN independent Q5 projection: invalid selected extent");
}

void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule = Q4GdnSimtR8T4Schedule;
    using Q5Schedule = Q5A16DirectSimtSchedule<1, 4, 4, 4, 10, kHidden, true>;
    // Q5 and Q4 publish disjoint rows; Q4 joins the Q5 producer at kernel exit.
    launch_q5_a16_direct_simt<Q5Schedule, true>(q5_linear_operands(x, value_z_weight),
                                                q5_projection_output(value, z),
                                                LinearIdentityEpilogue{}, stream);
    launch_q4_a16_simt<Q4Schedule, false, true, true>(
        q4_linear_operands(x, qk_weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(qk.data),
                                static_cast<std::int64_t>(qk.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void launch_q4_q5_gdn_input_independent(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    select_q4_projection(x.ne[1])(x, qk_weight, qk, stream);
    select_q5_projection(x.ne[1])(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
