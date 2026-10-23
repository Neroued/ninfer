#pragma once

#include "models/common/frontend/media.h"
#include "models/common/frontend/digest.h"
#include "ninfer/types.h"

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
    std::span<const std::uint32_t> execution_frontiers;
};

bool input_prefix_equal(InputPrefixView left, InputPrefixView right, std::size_t frontier);
bool complete_input_frontier(InputPrefixView input, std::size_t frontier);

// Shortlisting only. Exact input comparison is still required before reusing model state.
class PrefixDigests {
public:
    explicit PrefixDigests(InputPrefixView input);
    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;
private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

} // namespace ninfer::models::glm5next
