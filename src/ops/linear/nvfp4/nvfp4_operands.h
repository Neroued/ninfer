#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail {
struct Nvfp4A16Operands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
    float alpha;
};

struct Nvfp4A4Operands {
    const std::uint8_t* x;
    const std::uint8_t* x_scales;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
    float alpha;
    Nvfp4ScaleLayout scale_layout;
};

inline Nvfp4A16Operands nvfp4_a16_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            w.n,
            w.k,
            x.ne[1],
            1.0f / w.weight_scale_divisor};
}

inline Nvfp4A4Operands nvfp4_a4_operands(const Weight& w, Nvfp4A4Workspace x, int tokens,
                                         Nvfp4ScaleLayout layout) {
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            w.n,
            w.k,
            tokens,
            1.0f / (w.input_scale_divisor * w.weight_scale_divisor),
            layout};
}

template <class Schedule, class Operands>
void validate_nvfp4_operands(const Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.rows % 128 || p.k <= 0 || p.k % 64 || p.tokens <= 0 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, 16) || !std::isfinite(p.alpha) || p.alpha <= 0)
        throw std::invalid_argument(
            "NVFP4 templates require complete M128/K64 tiles and aligned operands");
    if constexpr (requires { p.x_scales; }) {
        if (!aligned(p.x_scales, 16))
            throw std::invalid_argument("NVFP4 A4 scales require 16-byte alignment");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("NVFP4 static K does not match operands");
    }
}
} // namespace ninfer::ops::detail
