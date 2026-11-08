#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <class Input = __nv_bfloat16>
struct Mxfp8A16Operands {
    static_assert(std::is_same_v<Input, __nv_bfloat16> || std::is_same_v<Input, float>);
    using InputType = Input;
    using ScaleType = std::uint8_t;
    const Input* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
};

struct Mxfp8A8Operands {
    const std::uint8_t* x;
    const std::uint8_t* x_scales;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
};

struct Mxfp8CompensatedA16Operands : Mxfp8A16Operands<> {
    const __nv_bfloat16* x_low;
};

inline Mxfp8A16Operands<> mxfp8_a16_operands(const Tensor& x, const Weight& weight) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            weight.n,
            weight.k,
            x.ne[1]};
}

} // namespace ninfer::ops::detail
