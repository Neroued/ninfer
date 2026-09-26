#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Q8LinearOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
    std::int32_t padded_k;
};

inline Q8LinearOperands q8_linear_operands(const Tensor& x, const Weight& weight) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            weight.n,
            weight.k,
            x.ne[1],
            weight.padded_shape[1]};
}

// Private template domain. Public shape admission remains in q8_dispatch.
inline void validate_q8_operands(const Q8LinearOperands& operands) {
    const auto aligned = [](const void* pointer, std::uintptr_t alignment) {
        return pointer && (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
    };
    if (operands.rows <= 0 || operands.tokens <= 0 || operands.k <= 0 ||
        operands.padded_k < operands.k || operands.padded_k % 128 != 0) {
        throw std::invalid_argument(
            "Q8 templates require positive N/K/T and padded K aligned to 128");
    }
    if (!aligned(operands.x, 16) || !aligned(operands.codes, 16) || !aligned(operands.scales, 16)) {
        throw std::invalid_argument("Q8 templates require aligned activation and weight planes");
    }
}

} // namespace ninfer::ops::detail
