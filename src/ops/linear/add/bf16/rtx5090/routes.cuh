#pragma once

#include "ops/linear/add/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

#include <type_traits>

namespace ninfer::ops::detail {

struct Bf16AddRoutes {
    template <class Schedule>
    static LinearAddLaunch gemv() {
        return bind<LinearResidualAddEpilogue,
                    launch_bf16_a16_gemv<Schedule, LinearBf16Output, LinearResidualAddEpilogue>>;
    }

    template <class Schedule, class Epilogue = LinearResidualAddEpilogue>
    static LinearAddLaunch sliced() {
        return bind<Epilogue, launch_bf16_a16_sliced_k_mma<Schedule, LinearBf16Output, Epilogue>>;
    }

    // The shared projection selector supplies its writeback type. LinearAdd
    // binds the same contraction to its own in-place residual epilogue.
    template <class Schedule, class = LinearIdentityEpilogue>
    static LinearAddLaunch tma() {
        return tma_with_epilogue<Schedule, Bf16ScalarResidualAddWarpStoreEpilogue>();
    }

    template <class Schedule, class Epilogue>
    static LinearAddLaunch tma_with_epilogue() {
        return bind<Epilogue, launch_bf16_a16_tma_mma<Schedule, 1, LinearBf16Output, Epilogue>>;
    }

private:
    template <class Epilogue, auto Launch>
    static void bind(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
        LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), w.n};
        const auto operands = bf16_a16_operands(x, w);
        if constexpr (std::is_same_v<Epilogue, LinearResidualAddEpilogue>)
            Launch(operands, output, Epilogue{{output.data, output.rows}}, stream);
        else
            Launch(operands, output, Epilogue{}, stream);
    }
};

} // namespace ninfer::ops::detail
