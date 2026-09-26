#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kSplitRow = 6144;
constexpr std::int32_t kHidden   = 5120;

using Q4AttnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

template <std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                            cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kHidden, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

void launch_q4_sliced_band(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("attention Q4 K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_sliced_exact<7>(x, weight, q, key, stream);
        return;
    case 8:
        launch_q4_sliced_exact<8>(x, weight, q, key, stream);
        return;
    case 9:
        launch_q4_sliced_exact<9>(x, weight, q, key, stream);
        return;
    case 10:
        launch_q4_sliced_exact<10>(x, weight, q, key, stream);
        return;
    case 11:
        launch_q4_sliced_exact<11>(x, weight, q, key, stream);
        return;
    case 12:
        launch_q4_sliced_exact<12>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        // K-split for the Q4 parent across the whole parent-split range. Complete-op measurement
        // (both parents launched, all four outputs, one graph, one probe run per column count):
        // 73.0-77.6 us at T=9..12 against 100.1-105.7 us for the row-split SIMT that R0 used there,
        // and 107.8-108.3 us for the grouped form the resolver switches to at T=13. The resolver
        // boundary at 13 is right for the grouped-vs-row-split question, but it hid this: the split
        // form with a K-split Q4 parent is 24-29% faster than both. T=2..6 keep the SIMT tile,
        // where a 16-wide K-split tile would waste more MMA work than it saves.
        launch_q4_sliced_band(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
        launch_q4_simt<Q4AttnSimtR8T4Schedule>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 split-output requires T in [1,12]");
    }
}

auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<kSplitRow>{{static_cast<__nv_bfloat16*>(first.data),
                                              std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)), 0},
                                             {static_cast<__nv_bfloat16*>(second.data),
                                              std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)),
                                              0}};
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(q5_linear_operands(x, weight),
                                                           q5_projection_output(gate, value),
                                                           LinearIdentityEpilogue{}, stream);
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, kHidden, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight),
                                        q5_projection_output(gate, value), LinearIdentityEpilogue{},
                                        stream);
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, gate, value, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, gate, value, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,9]");
    }
}

template <int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    using Schedule = Q5A16SimtSchedule<8, ColsPerTile, 1, 16, 2, Cache::ca, 1>;
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(gate, value),
                                 LinearIdentityEpilogue{}, stream);
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 9) {
        // Split4: one CTA owns one output row, its four warps split the K dimension and reduce
        // their partial sums through shared memory, with the column count as a compile-time
        // template argument so the kernel covers exactly the live columns. The fused projections
        // use the same shape up to 6, so the Q5 parent has one mechanism from 2 to 9.
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 12) {
        // c4 SIMT: one output row per warp, up to four columns per column tile, with the quantized
        // weight planes staged in shared memory and activations read from the input tensor.
        // Retained in this interval on complete-Op measurements; both shapes are legal at every
        // count in [2,12].
        launch_q5_simt<4>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,12]");
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail
