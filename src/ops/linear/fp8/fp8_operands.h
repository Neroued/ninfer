#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
struct Fp8A16Operands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
    int rows, k, tokens;
};

struct Fp8A8Operands {
    const std::uint8_t* x;
    const float* x_scales;
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
    int rows, k, tokens;
};

inline Fp8A16Operands fp8_a16_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const __nv_bfloat16*>(w.scales),
            w.n,
            w.k,
            x.ne[1]};
}

inline Fp8A8Operands fp8_a8_operands(const Weight& w, Fp8A8Workspace x, int tokens) {
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const __nv_bfloat16*>(w.scales),
            w.n,
            w.k,
            tokens};
}

// Private template admission is independent of the public finite shape registry.
template <class Schedule, class Operands>
inline void validate_fp8_operands(const Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.k % 32 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, 2))
        throw std::invalid_argument("FP8 templates require positive N/K/T and aligned operands");
    if constexpr (requires { p.x_scales; }) {
        if (!aligned(p.x_scales, 4) || !aligned(p.scales, 4))
            throw std::invalid_argument("FP8 A8 requires scales aligned to four bytes");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("FP8 template static K does not match operands");
    }
}
} // namespace ninfer::ops::detail
