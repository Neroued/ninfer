#pragma once

#include "ninfer/ops/context_bucket.h"

#include <algorithm>

namespace ninfer::ops::detail {

[[nodiscard]] inline OpGraphProfile make_op_graph_profile(std::span<const std::uint64_t> words) {
    OpGraphProfile profile;
    if (words.size() > profile.words_.size()) {
        throw std::invalid_argument("Op graph profile has too many words");
    }
    std::copy(words.begin(), words.end(), profile.words_.begin());
    return profile;
}

} // namespace ninfer::ops::detail
