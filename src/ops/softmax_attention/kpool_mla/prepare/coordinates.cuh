#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/arguments.h"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::kpool {

inline void set_window(PrepareArguments& a, std::int64_t begin, int columns) {
    a.first_row        = begin / a.tokens;
    a.first_column     = begin - std::int64_t(a.first_row) * a.tokens;
    a.first_span       = a.tokens - a.first_column;
    a.last_row         = (begin + columns - 1) / a.tokens;
    a.last_stop        = begin + columns - std::int64_t(a.last_row) * a.tokens;
    const auto divisor = common::FastDivisorU31::make(a.tokens);
    a.token_multiplier = divisor.multiplier;
    a.token_shift      = divisor.shift;
    a.stage_columns    = columns;
}

// Only the bounded window remainder needs division. The full flat position can exceed I32;
// it is never narrowed, and no Device 64-bit division is needed for batch addressing.
__device__ __forceinline__ void coordinate(const PrepareArguments& a, int column, int& b, int& t) {
    if (column < a.first_span) {
        b = a.first_row;
        t = a.first_column + column;
    } else {
        const auto delta = std::uint32_t(column - a.first_span);
        const auto row   = (__umulhi(delta, a.token_multiplier) + delta) >> a.token_shift;
        b                = a.first_row + 1 + row;
        t                = delta - row * a.tokens;
    }
}
} // namespace ninfer::ops::detail::kpool
