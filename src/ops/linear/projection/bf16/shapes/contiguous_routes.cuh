#pragma once

#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {

struct Bf16ContiguousRoutes {
    template <class S>
    static Bf16Launch gemv() {
        return launch_bf16_gemv<S>;
    }

    template <class S>
    static Bf16Launch sliced() {
        return launch_bf16_sliced_k_mma<S>;
    }

    template <class S>
    static Bf16Launch mma() {
        return launch_bf16_mma<S>;
    }

    template <class S>
    static Bf16Launch tma() {
        return launch_bf16_tma_mma<S>;
    }
};

} // namespace ninfer::ops::detail
