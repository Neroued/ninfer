#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Bf16A16Operands {
    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    int rows, k, tokens;
};

inline Bf16A16Operands bf16_a16_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(w.qdata),
            w.n, w.k, x.ne[1]};
}

// Private template admission is independent of the public finite shape registry.
template <class Schedule>
void validate_bf16_operands(const Bf16A16Operands& p) {
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.k % 8 || !aligned(p.x) || !aligned(p.weight))
        throw std::invalid_argument("BF16 templates require positive N/K/T and aligned operands");
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("BF16 template static K does not match operands");
    }
}

} // namespace ninfer::ops::detail
