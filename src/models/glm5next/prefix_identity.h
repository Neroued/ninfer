#pragma once

#include "models/common/frontend/digest.h"
#include "models/common/frontend/media.h"
#include "ninfer/types.h"

#include <span>

namespace ninfer::models::glm5next {

struct MediaInputIdentity {
    models::frontend::VisionItem item;
    // Binds preprocessing parameters and the realized patch representation, independently of rank.
    models::frontend::Sha256Digest preparation_digest{};
};

// Positions are token ordinals. Views borrow immutable input; placement and cache policy are
// absent.
struct InputPrefixView {
    std::span<const TokenId> tokens;
    std::span<const MediaInputIdentity> media;
};

bool complete_input_frontier(InputPrefixView input, std::size_t frontier);

} // namespace ninfer::models::glm5next
