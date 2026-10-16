#include "ops/linear/projection/q8/rtx5090/shapes.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"

#include "ops/linear/projection/q8/rtx5090/shapes/n12288_k2560.cuh"
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void packed_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, weight),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void packed_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void packed_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void packed_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}

struct Routes {
    template <class S>
    static Q8Launch gemv() {
        return packed_gemv<S>;
    }

    template <class S>
    static Q8Launch sliced() {
        return packed_sliced<S>;
    }

    template <class S>
    static Q8Launch grouped() {
        return packed_grouped<S>;
    }

    template <class S>
    static Q8Launch mma() {
        if constexpr (std::is_same_v<S, q8_instances::MmaR32T64>)
            return launch_q8_a16_mma_r32_t64;
        else if constexpr (std::is_same_v<S, q8_instances::MmaR48T64>)
            return launch_q8_a16_mma_r48_t64;
        else
            return packed_mma<S>;
    }
};

} // namespace

Q8Launch select_q8_n12288_k2560(std::int32_t tokens) {
    return q8_n12288_k2560::select(tokens, Routes{});
}

} // namespace ninfer::ops::detail
