#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision             = false;
    bool mtp                = false;
    bool optimized_proposal = false;

    bool operator==(const StartupFeatures&) const = default;
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    const bool mtp = options.speculative.draft_tokens != 0;
    return StartupFeatures{
        .vision             = options.enable_vision,
        .mtp                = mtp,
        .optimized_proposal = mtp && options.speculative.proposal_head == ProposalHead::Optimized,
    };
}

} // namespace ninfer::targets::qwen3_6
