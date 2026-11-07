#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <class Input = __nv_bfloat16>
struct Fp8BlockA16Operands {
    static_assert(std::is_same_v<Input, __nv_bfloat16> || std::is_same_v<Input, float>);
    using InputType = Input;
    using ScaleType = float;
    const Input* x;
    const std::uint8_t* codes;
    const float* scales;
    int rows, k, tokens;
};

struct Fp8BlockCompensatedA16Operands : Fp8BlockA16Operands<> {
    const __nv_bfloat16* x_low;
};

inline Fp8BlockA16Operands<> fp8_block_a16_operands(const Tensor& x, const Weight& weight) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const float*>(weight.scales),
            weight.n,
            weight.k,
            x.ne[1]};
}

} // namespace ninfer::ops::detail
