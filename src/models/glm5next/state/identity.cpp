#include "models/glm5next/state/identity.h"

#include <algorithm>

namespace ninfer::models::glm5next {
InputIdentity::InputIdentity(const PreparedPromptData& p, std::uint32_t domain)
    : tokens(p.token_ids), media(p.identity.media),
      execution_frontiers(p.identity.execution_frontiers), digests(p.prefix_view()), tag(domain) {}

void InputIdentity::reserve(std::size_t n) {
    tokens.reserve(n);
    digests.reserve(n);
}

void InputIdentity::prepare_generated(std::size_t count, std::optional<std::uint32_t> split) {
    if (split && (!*split || *split > count))
        throw std::invalid_argument("GLM generated boundary exceeds input");
    reserve(tokens.size() + count);
    if (split) execution_frontiers.reserve(execution_frontiers.size() + 1);
}

void InputIdentity::append_generated(std::span<const TokenId> ids,
                                     std::optional<std::uint32_t> split) {
    prepare_generated(ids.size(), split);
    const auto begin = tokens.size();
    digests.append_generated(ids, split);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    if (split) execution_frontiers.push_back(begin + *split);
}

void InputIdentity::truncate(std::uint32_t frontier) {
    if (!complete_input_frontier(view(), frontier))
        throw std::invalid_argument("GLM capture intersects media");
    tokens.resize(frontier);
    digests.truncate(frontier);
    std::erase_if(media,
                  [&](const auto& m) { return m.item.token_spans.front().begin >= frontier; });
    std::erase_if(execution_frontiers, [&](auto f) { return f > frontier; });
}
} // namespace ninfer::models::glm5next
