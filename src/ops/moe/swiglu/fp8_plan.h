#pragma once

#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

#include <algorithm>
#include <array>
#include <limits>

namespace ninfer::ops::detail::swiglu_ffn {

inline bool scaled_fp8_format(QType format) {
    return format == QType::MXFP8 || format == QType::FP8_E4M3FN_BLOCK128X128_FP32;
}

enum class ScaledFp8Product { Float, Compensated, Mxfp8 };

struct ScaledFp8Plan {
    int last_token;
    ScaledFp8Product product;
};

inline std::array<ScaledFp8Plan, 2> scaled_fp8_plans(const SwiGLUProfile& profile) {
    const bool a8 = profile.gate_up_type == QType::MXFP8 && allows_a8(profile.gate_up_policy) &&
                    allows_a8(profile.down_policy);
    return {{{4, ScaledFp8Product::Float},
             {std::numeric_limits<int>::max(),
              a8 ? ScaledFp8Product::Mxfp8 : ScaledFp8Product::Compensated}}};
}

inline ScaledFp8Plan select_scaled_fp8(const SwiGLUProfile& profile, int tokens) {
    const auto plans = scaled_fp8_plans(profile);
    return tokens <= plans.front().last_token ? plans.front() : plans.back();
}

struct ScaledFp8Storage {
    void* product = nullptr;
    Mxfp8A8Workspace input_a8{}, product_a8{};
};

template <class Arena>
ScaledFp8Storage allocate_scaled_fp8(Arena& arena, const SwiGLUProfile& profile, int tokens) {
    ScaledFp8Storage out;
    if (select_scaled_fp8(profile, tokens).product == ScaledFp8Product::Mxfp8) {
        out.input_a8   = allocate_mxfp8_a8_workspace(arena, tokens, profile.hidden);
        out.product_a8 = allocate_mxfp8_a8_workspace(arena, tokens, profile.intermediate);
    } else {
        out.product =
            arena.alloc_bytes(std::size_t(profile.intermediate) * tokens * sizeof(float)).data;
    }
    return out;
}

inline std::size_t scaled_fp8_workspace_bytes(const SwiGLUProfile& profile, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scaled_fp8(layout, profile, tokens);
    return layout.peak_bytes(1);
}

inline std::size_t scaled_fp8_workspace_capacity_bytes(const SwiGLUProfile& profile, int first,
                                                       int last) {
    auto result = std::max(scaled_fp8_workspace_bytes(profile, first),
                           scaled_fp8_workspace_bytes(profile, last));
    for (const auto& plan : scaled_fp8_plans(profile))
        if (first <= plan.last_token && last >= plan.last_token)
            result = std::max(result, scaled_fp8_workspace_bytes(profile, plan.last_token));
    return result;
}

} // namespace ninfer::ops::detail::swiglu_ffn
