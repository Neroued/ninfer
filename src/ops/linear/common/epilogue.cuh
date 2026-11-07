#pragma once

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }
};

struct LinearBiasEpilogue {
    const __nv_bfloat16* bias;

    __device__ __forceinline__ float apply(int row, int, float value) const {
        return value + __bfloat162float(bias[row]);
    }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return value + residual.load(row, token);
    }
};

// Row consumers run in the single thread owning the fully reduced row. They
// may publish fused Op state, but do not introduce warp/CTA synchronization.
template <class Output, class Epilogue, int Tokens>
__device__ __forceinline__ void
linear_finish_row(const Output& output, const Epilogue& epilogue, int row, int token_begin,
                  const float (&values)[Tokens], int active_tokens) {
    if constexpr (requires {
                      epilogue.apply_row(output, row, token_begin, values, active_tokens);
                  }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens);
    } else {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < active_tokens) {
                const int global_token = token_begin + token;
                output.store(row, global_token, epilogue.apply(row, global_token, values[token]));
            }
        }
    }
}

template <bool FullTokens, class Output, class Epilogue, bool FullRows = true>
__device__ __forceinline__ void
linear_finish_mma_fragment(const Output& output, const Epilogue& epilogue, int row, int token,
                           float4 value, int rows, int tokens) {
    if constexpr (requires { epilogue.store_fragment(output, row, token, value, rows, tokens); }) {
        static_assert(FullRows, "row-tail MMA requires a scalar epilogue");
        // A fragment consumer receives the atom origin and owns its token-tail predicates.
        epilogue.store_fragment(output, row, token, value, rows, tokens);
    } else {
        if (FullTokens || token < tokens) {
            if (FullRows || row < rows)
                output.store(row, token, epilogue.apply(row, token, value.x));
            if (FullRows || row + 8 < rows)
                output.store(row + 8, token, epilogue.apply(row + 8, token, value.z));
        }
        if (FullTokens || token + 1 < tokens) {
            if (FullRows || row < rows)
                output.store(row, token + 1, epilogue.apply(row, token + 1, value.y));
            if (FullRows || row + 8 < rows)
                output.store(row + 8, token + 1, epilogue.apply(row + 8, token + 1, value.w));
        }
    }
}

template <class Schedule, class Epilogue>
inline constexpr int linear_epilogue_bytes = [] {
    if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
        return static_cast<int>(Epilogue::template kSharedBytes<Schedule>);
    else
        return 0;
}();

template <class Schedule, class Epilogue>
inline constexpr int linear_mma_shared_bytes =
    Schedule::kSharedBytes > linear_epilogue_bytes<Schedule, Epilogue>
        ? Schedule::kSharedBytes
        : linear_epilogue_bytes<Schedule, Epilogue>;

} // namespace ninfer::ops::detail
