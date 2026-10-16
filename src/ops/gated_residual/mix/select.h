#pragma once

#include "ops/gated_residual/mix/operands.h"

namespace ninfer::ops::detail::hc::mix {
enum class Down {
    Gemv,
    Simt,
    Split8x16,
    Split8x32,
    TmaSplit2,
    TmaSplit4,
    Tma80,
    Tma96Split4,
    Tma96Split8
};
enum class Up {
    Folded16x16,
    Folded32x32,
    TmaFolded16x64,
    TmaFolded32x64,
};

struct Plan {
    Down down;
    Up up;
};

constexpr int split_factor(Down down) {
    switch (down) {
    case Down::Split8x16:
    case Down::Split8x32:
    case Down::Tma96Split8:
        return 8;
    case Down::TmaSplit2:
        return 2;
    case Down::TmaSplit4:
    case Down::Tma96Split4:
        return 4;
    default:
        return 1;
    }
}

Plan select_plan(Kind kind, std::int32_t columns);
} // namespace ninfer::ops::detail::hc::mix
