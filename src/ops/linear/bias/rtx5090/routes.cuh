#pragma once

#include "ops/linear/bias/bf16_launch.cuh"
#include "ops/linear/bias/dispatch.h"

#include <type_traits>

namespace ninfer::ops::detail::linear_bias {

struct Bf16BiasRoutes {
    template <class S>
    static Launch gemv() {
        return linear_bias::gemv<S>;
    }

    template <class S>
    static Launch sliced() {
        return linear_bias::sliced<S>;
    }

    template <class S>
    static Launch mma() {
        return linear_bias::mma<S>;
    }

    template <class S, class Epilogue = LinearIdentityEpilogue>
    static Launch tma() {
        if constexpr (std::is_same_v<Epilogue, LinearIdentityEpilogue>) {
            return linear_bias::tma<S>;
        } else {
            static_assert(std::is_same_v<Epilogue, Bf16MmaWarpStoreEpilogue> ||
                          std::is_same_v<Epilogue, Bf16TmaVectorEpilogue>);
            // Both pure stores use complete row tiles. The bias warp store accepts
            // naturally aligned bias and adds it before packing the BF16 output.
            return linear_bias::warp<S>;
        }
    }
};

} // namespace ninfer::ops::detail::linear_bias
