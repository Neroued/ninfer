#pragma once

#include "models/common/frontend/digest.h"
#include "models/common/frontend/media.h"
#include "ninfer/types.h"

#include <span>

namespace ninfer::models::inkling2 {

struct MediaInputIdentity {
    models::frontend::VisionItem item;
    models::frontend::Sha256Digest preparation_digest{};
};

struct InputPrefixView {
    std::span<const TokenId> tokens;
    std::span<const MediaInputIdentity> media;
};

bool complete_input_frontier(InputPrefixView input, std::size_t frontier);

} // namespace ninfer::models::inkling2
