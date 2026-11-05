#pragma once

#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace ninfer::ops::detail::swiglu_ffn {

inline bool scaled_fp8_format(QType format) {
    return format == QType::MXFP8 || format == QType::FP8_E4M3FN_BLOCK128X128_FP32;
}

enum class ScaledFp8Product { Float, Compensated, Mxfp8 };

enum class ScaledFp8Compute { Simt, Bf16CompensatedDown, Mxfp8 };

enum class ScaledFp8Kernel {
    Default,
    DenseFloat,
    DenseCompensated,
    DensePaddedSmall,
    DenseSmall,
    DenseMma,
    DenseTransition,
    DenseBulk
};

struct ScaledFp8Plan {
    int last_token;
    ScaledFp8Product product;
    ScaledFp8Kernel kernel = ScaledFp8Kernel::Default;
    int token_alignment    = 1;

    ScaledFp8Compute compute() const {
        if (product == ScaledFp8Product::Mxfp8) return ScaledFp8Compute::Mxfp8;
        if (product == ScaledFp8Product::Compensated || kernel != ScaledFp8Kernel::Default)
            return ScaledFp8Compute::Bf16CompensatedDown;
        return ScaledFp8Compute::Simt;
    }

    int storage_tokens(int tokens) const { return mxfp8_a8_padded_tokens(tokens, token_alignment); }
};

inline std::span<const ScaledFp8Plan> scaled_fp8_plans(const SwiGLUProfile& profile,
                                                       bool routed_join = false) {
    const bool a8 = profile.gate_up_type == QType::MXFP8 && allows_a8(profile.gate_up_policy) &&
                    allows_a8(profile.down_policy);
    constexpr int all = std::numeric_limits<int>::max();
    static constexpr std::array native{ScaledFp8Plan{4, ScaledFp8Product::Float},
                                       ScaledFp8Plan{all, ScaledFp8Product::Compensated}};
    static constexpr std::array quantized{ScaledFp8Plan{4, ScaledFp8Product::Float},
                                          ScaledFp8Plan{all, ScaledFp8Product::Mxfp8}};
    static constexpr std::array dense{
        ScaledFp8Plan{8, ScaledFp8Product::Float, ScaledFp8Kernel::DenseFloat},
        ScaledFp8Plan{16, ScaledFp8Product::Compensated, ScaledFp8Kernel::DenseCompensated},
        ScaledFp8Plan{32, ScaledFp8Product::Float, ScaledFp8Kernel::DenseFloat},
        ScaledFp8Plan{64, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::DensePaddedSmall, 64},
        ScaledFp8Plan{96, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::DenseSmall},
        ScaledFp8Plan{128, ScaledFp8Product::Mxfp8},
        ScaledFp8Plan{192, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::DenseMma, 64},
        ScaledFp8Plan{384, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::DenseTransition, 128},
        ScaledFp8Plan{all / 128 * 128, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::DenseBulk, 128},
        ScaledFp8Plan{all, ScaledFp8Product::Mxfp8}};
    static constexpr auto dense_join = [] {
        auto plans = dense;
        // The routed FP32 read favors A8 contraction earlier than plain BF16 output.
        plans[2].last_token      = 24;
        plans[3].token_alignment = 32;
        return plans;
    }();
    if (a8 && profile.hidden == 4096 && profile.intermediate == 6144)
        return routed_join ? std::span<const ScaledFp8Plan>(dense_join)
                           : std::span<const ScaledFp8Plan>(dense);
    return a8 ? std::span<const ScaledFp8Plan>(quantized) : std::span<const ScaledFp8Plan>(native);
}

inline ScaledFp8Plan select_scaled_fp8(const SwiGLUProfile& profile, int tokens,
                                       bool routed_join = false) {
    for (const auto& plan : scaled_fp8_plans(profile, routed_join))
        if (tokens <= plan.last_token) return plan;
    throw std::invalid_argument("scaled FP8 FFN token count exceeds int range");
}

// Necessary staging writes and reads, excluding alignment gaps and CTA reuse.
inline std::size_t scaled_fp8_workspace_traffic_bytes(const SwiGLUProfile& profile, int tokens,
                                                      bool routed_join = false) {
    const auto plan = select_scaled_fp8(profile, tokens, routed_join);
    if (plan.product == ScaledFp8Product::Mxfp8)
        return std::size_t(2) * plan.storage_tokens(tokens) *
               (profile.hidden + profile.hidden / 32 + profile.intermediate +
                profile.intermediate / 32);
    return std::size_t(2) * profile.intermediate * tokens * sizeof(float);
}

struct ScaledFp8Storage {
    void* product = nullptr;
    Mxfp8A8Workspace input_a8{}, product_a8{};
};

template <class Arena>
ScaledFp8Storage allocate_scaled_fp8(Arena& arena, const SwiGLUProfile& profile, int tokens,
                                     bool routed_join = false) {
    ScaledFp8Storage out;
    const auto plan = select_scaled_fp8(profile, tokens, routed_join);
    if (plan.product == ScaledFp8Product::Mxfp8) {
        const int physical = plan.storage_tokens(tokens);
        out.input_a8       = allocate_mxfp8_a8_workspace(arena, physical, profile.hidden);
        out.product_a8     = allocate_mxfp8_a8_workspace(arena, physical, profile.intermediate);
    } else {
        out.product =
            arena.alloc_bytes(std::size_t(profile.intermediate) * tokens * sizeof(float)).data;
    }
    return out;
}

inline std::size_t scaled_fp8_workspace_bytes(const SwiGLUProfile& profile, int tokens) {
    std::size_t result = 0;
    for (bool joined : {false, true}) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_scaled_fp8(layout, profile, tokens, joined);
        result = std::max(result, layout.peak_bytes(1));
    }
    return result;
}

inline std::size_t scaled_fp8_workspace_capacity_bytes(const SwiGLUProfile& profile, int first,
                                                       int last) {
    auto result = std::max(scaled_fp8_workspace_bytes(profile, first),
                           scaled_fp8_workspace_bytes(profile, last));
    for (bool joined : {false, true})
        for (const auto& plan : scaled_fp8_plans(profile, joined))
            if (first <= plan.last_token && last >= plan.last_token)
                result = std::max(result, scaled_fp8_workspace_bytes(profile, plan.last_token));
    return result;
}

} // namespace ninfer::ops::detail::swiglu_ffn
