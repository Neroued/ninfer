#pragma once

// The column slices of a tp 2 prefill chunk whose row-parallel all-reduces are pipelined slice by
// slice (Qwen4Forward::run_plain, execution/qwen4_forward/plain.h), and the split of the half-chunk
// overlap (prefill_overlap_split). Header-only and free of CUDA, so the plan is unit-tested on the
// host (tests/test_tp2_prefill_slices.cpp).
//
// The sliced schedule keeps every mixer core (norms, input projections, attention, GDN) on the
// whole chunk and cuts only the column-local tail of each block: the output projection with its
// all-reduce, the post-attention norm and the FFN. A slice is therefore free of the half-chunk
// overlap's limits (both halves > 256 columns for the attention family, no multimodal RoPE), so
// it serves chunks under 768 columns, multimodal chunks and batched-feature taps. The plan:
//
//   width  128 * max(1, len / 512) (about four slices, as kurcontko/ninfer-tp2 d0925e24), or the
//          override (a positive multiple of 128), raised by 128 until at most kMaxCount slices;
//   slices [s * width, (s + 1) * width), the last one takes the rest; a rest under kMinTail
//          columns joins the slice before it, so no slice is ever narrow enough to change the
//          linear route family (the A16 routes, whose K partition differs, start under 17);
//   count  >= 2, else the plan is empty (count 0) and the chunk runs as one call.

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace ninfer::models::qwen3_5::execution {

inline constexpr std::int32_t kPrefillSliceAlign    = 128;
inline constexpr std::int32_t kPrefillSliceMaxCount = 8; // ops::HostExchange::kSumLanes
inline constexpr std::int32_t kPrefillSliceMinTail  = 64;

// NINFER_TP_PREFILL_SLICES: "0" off (one call per chunk where the halves do not apply), "force"
// slices even where the half-chunk overlap would run (bit and speed tests), anything else (unset)
// auto: slices where the halves do not apply.
enum class PrefillSlicesMode : int { Off = 0, Auto = 1, Force = 2 };

[[nodiscard]] constexpr PrefillSlicesMode prefill_slices_mode(std::string_view env) noexcept {
    if (env == "0") { return PrefillSlicesMode::Off; }
    if (env == "force") { return PrefillSlicesMode::Force; }
    return PrefillSlicesMode::Auto;
}

struct PrefillSlicePlan {
    std::int32_t width = 0;
    std::int32_t count = 0;

    [[nodiscard]] constexpr bool usable() const noexcept { return count >= 2; }

    [[nodiscard]] constexpr std::int32_t first(std::int32_t slice) const noexcept {
        return slice * width;
    }

    [[nodiscard]] constexpr std::int32_t columns(std::int32_t slice,
                                                 std::int32_t len) const noexcept {
        return slice + 1 < count ? width : len - slice * width;
    }
};

// A valid width override: 0 (none) or a positive multiple of kPrefillSliceAlign.
[[nodiscard]] constexpr bool prefill_slice_width_valid(std::int32_t width) noexcept {
    return width == 0 || (width > 0 && width % kPrefillSliceAlign == 0);
}

[[nodiscard]] constexpr PrefillSlicePlan prefill_slice_plan(std::int32_t len,
                                                            std::int32_t width_override) noexcept {
    PrefillSlicePlan plan;
    if (len < 2 * kPrefillSliceAlign || !prefill_slice_width_valid(width_override)) { return plan; }
    std::int32_t width = width_override;
    if (width == 0) {
        const std::int32_t quarters = len / (4 * kPrefillSliceAlign);
        width                       = kPrefillSliceAlign * (quarters > 1 ? quarters : 1);
    }
    std::int32_t count = 0;
    for (;; width += kPrefillSliceAlign) {
        const std::int32_t whole = len / width;
        const std::int32_t rest  = len - whole * width;
        count                    = whole + (rest >= kPrefillSliceMinTail ? 1 : 0);
        if (count <= kPrefillSliceMaxCount) { break; }
    }
    if (count < 2) { return plan; }
    plan.width = width;
    plan.count = count;
    return plan;
}

// NINFER_TP_PREFILL_OVERLAP=0 turns the half-chunk overlap and the slices off; read once.
inline bool prefill_overlap_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TP_PREFILL_OVERLAP");
        return value == nullptr || std::string_view(value) != "0";
    }();
    return enabled;
}

// The sliced prefill's mode (NINFER_TP_PREFILL_SLICES; Off without the overlap) and width override
// (NINFER_TP_PREFILL_SLICE_TOKENS, a positive multiple of 128; unset or 0: the plan's own width),
// read once. Read by the plain residual (qwen4_forward/plain.h) and the startup transport status.
inline PrefillSlicesMode prefill_slices_mode_env() {
    static const PrefillSlicesMode mode = [] {
        if (!prefill_overlap_enabled()) { return PrefillSlicesMode::Off; }
        const char* value = std::getenv("NINFER_TP_PREFILL_SLICES");
        return prefill_slices_mode(value != nullptr ? value : "");
    }();
    return mode;
}

inline std::int32_t prefill_slice_tokens_env() {
    static const std::int32_t tokens = [] {
        const char* value = std::getenv("NINFER_TP_PREFILL_SLICE_TOKENS");
        if (value == nullptr || *value == '\0') { return std::int32_t{0}; }
        char* end         = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (*end != '\0' || parsed < 0 || parsed > (1 << 20) ||
            !prefill_slice_width_valid(static_cast<std::int32_t>(parsed))) {
            throw std::invalid_argument(
                "NINFER_TP_PREFILL_SLICE_TOKENS must be 0 or a positive multiple of 128");
        }
        return static_cast<std::int32_t>(parsed);
    }();
    return tokens;
}

// The half-chunk overlap (Qwen4Forward::run_plain): the width of the first half of a `len`-column
// text chunk, or 0 to run the chunk without it. A multiple of 128, so the GDN chunk (16 tokens),
// the attention query tiles and the activation-scale tiles fall on the same column boundaries as in
// one call over the chunk; both halves wider than 256 columns, so their causal attention takes the
// tiled family a whole chunk takes (the grouped family serves widths up to 256).
inline int prefill_overlap_split(int len) {
    constexpr int kAlign = 128, kMinHalf = 384;
    if (!prefill_overlap_enabled()) { return 0; }
    const int split = len / 2 / kAlign * kAlign;
    return split >= kMinHalf && len - split >= kMinHalf ? split : 0;
}

} // namespace ninfer::models::qwen3_5::execution
