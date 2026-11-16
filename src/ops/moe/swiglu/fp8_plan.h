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

enum class ScaledFp8Product { Float, Compensated, Mxfp8, ScaledF16 };

enum class ScaledFp8Compute { Simt, Bf16CompensatedDown, Mxfp8, Bf16F16 };

enum class ScaledFp8Kernel {
    Default,
    DenseFloat,
    DenseCompensated,
    DensePaddedSmall,
    DenseSmall,
    DenseMma,
    DenseTransition,
    DenseBulk,
    BlockFloat,
    BlockCompensated,
    BlockScaledF16Decode,
    BlockScaledF16Small,
    BlockScaledF16Medium,
    BlockScaledF16Wide,
    BlockScaledF16Bulk,
    MtpFloat,
    MtpSmall,
    MtpPadded,
    MtpTransition,
    MtpBulkMedium,
    MtpBulk,
    MtpLarge
};

struct ScaledFp8Plan {
    int last_token;
    ScaledFp8Product product;
    ScaledFp8Kernel kernel = ScaledFp8Kernel::Default;
    int token_alignment    = 1;
    int product_group      = 32;

    ScaledFp8Compute compute() const {
        if (product == ScaledFp8Product::Mxfp8) return ScaledFp8Compute::Mxfp8;
        if (product == ScaledFp8Product::ScaledF16) return ScaledFp8Compute::Bf16F16;
        if (product == ScaledFp8Product::Compensated || kernel != ScaledFp8Kernel::Default)
            return ScaledFp8Compute::Bf16CompensatedDown;
        return ScaledFp8Compute::Simt;
    }

    int storage_tokens(int tokens) const { return mxfp8_a8_padded_tokens(tokens, token_alignment); }
};

constexpr int kScaledF16DecodeDownTokens = 16;

inline int scaled_f16_decode_gate_up_tokens(int tokens) { return tokens <= 24 ? 8 : 16; }

inline std::span<const ScaledFp8Plan> scaled_fp8_plans(const SwiGLUProfile& profile,
                                                       bool routed_join = false) {
    const bool a8 = profile.gate_up_type == QType::MXFP8 && allows_a8(profile.gate_up_policy) &&
                    allows_a8(profile.down_policy);
    constexpr int all = std::numeric_limits<int>::max();
    static constexpr std::array native{ScaledFp8Plan{4, ScaledFp8Product::Float},
                                       ScaledFp8Plan{all, ScaledFp8Product::Compensated}};
    static constexpr std::array quantized{ScaledFp8Plan{4, ScaledFp8Product::Float},
                                          ScaledFp8Plan{all, ScaledFp8Product::Mxfp8}};
    static constexpr std::array mtp{
        ScaledFp8Plan{1, ScaledFp8Product::Float, ScaledFp8Kernel::MtpFloat},
        ScaledFp8Plan{8, ScaledFp8Product::Float, ScaledFp8Kernel::MtpFloat},
        ScaledFp8Plan{64, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpSmall, 96},
        ScaledFp8Plan{96, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpSmall, 32},
        ScaledFp8Plan{192, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpPadded, 192},
        ScaledFp8Plan{384, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpTransition, 128},
        ScaledFp8Plan{512, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpBulkMedium, 128},
        ScaledFp8Plan{2048, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpBulk, 128},
        ScaledFp8Plan{all / 128 * 128, ScaledFp8Product::Mxfp8, ScaledFp8Kernel::MtpLarge, 128},
        ScaledFp8Plan{all, ScaledFp8Product::Mxfp8}};
    static constexpr auto mtp_join = [] {
        auto plans = mtp;
        plans[0]   = ScaledFp8Plan{1, ScaledFp8Product::Float};
        return plans;
    }();
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
    static constexpr std::array block{
        ScaledFp8Plan{2, ScaledFp8Product::Float},
        ScaledFp8Plan{8, ScaledFp8Product::Float, ScaledFp8Kernel::BlockFloat},
        ScaledFp8Plan{16, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Decode, 1,
                      16},
        // The existing GU tile also publishes zero products for its unused columns.
        // An eight-column physical suffix avoids the down stage's thin-tail penalty.
        ScaledFp8Plan{24, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Decode, 8,
                      16},
        ScaledFp8Plan{32, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Decode, 1,
                      16},
        ScaledFp8Plan{48, ScaledFp8Product::Compensated, ScaledFp8Kernel::BlockCompensated},
        ScaledFp8Plan{128, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Small, 32,
                      32},
        ScaledFp8Plan{192, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Medium, 64,
                      32},
        ScaledFp8Plan{384, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Wide, 128,
                      128},
        ScaledFp8Plan{2048, ScaledFp8Product::ScaledF16, ScaledFp8Kernel::BlockScaledF16Bulk, 128,
                      64},
        ScaledFp8Plan{all / 128 * 128, ScaledFp8Product::ScaledF16,
                      ScaledFp8Kernel::BlockScaledF16Wide, 128, 128},
        ScaledFp8Plan{all, ScaledFp8Product::Compensated}};
    if (profile.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_FP32) return block;
    if (a8 && profile.hidden == 4096 && profile.intermediate == 6144)
        return routed_join ? std::span<const ScaledFp8Plan>(dense_join)
                           : std::span<const ScaledFp8Plan>(dense);
    if (a8 && profile.hidden == 4096 && profile.intermediate == 1024)
        return routed_join ? std::span<const ScaledFp8Plan>(mtp_join)
                           : std::span<const ScaledFp8Plan>(mtp);
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
    if (plan.product == ScaledFp8Product::ScaledF16)
        return std::size_t(2) * plan.storage_tokens(tokens) *
               (profile.intermediate * 2 + profile.intermediate / plan.product_group * 4);
    return std::size_t(2) * profile.intermediate * tokens * sizeof(float);
}

struct ScaledFp8Storage {
    void* product         = nullptr;
    float* product_scales = nullptr;
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
    } else if (plan.product == ScaledFp8Product::ScaledF16) {
        const int physical = plan.storage_tokens(tokens);
        out.product = arena.alloc_bytes(std::size_t(profile.intermediate) * physical * 2).data;
        out.product_scales = static_cast<float*>(
            arena.alloc_bytes(std::size_t(profile.intermediate / plan.product_group) * physical * 4)
                .data);
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
