#include "models/glm5next/state/identity.h"

#include <algorithm>

namespace ninfer::models::glm5next {
InputIdentity::InputIdentity(const PreparedPromptData& p, std::uint32_t domain)
    : tokens(p.token_ids), media(p.identity.media), digests(p.prefix_view()), tag(domain) {}

void InputIdentity::reserve(std::size_t n) {
    tokens.reserve(n);
    digests.reserve(n);
}

void InputIdentity::prepare_generated(std::size_t count) { reserve(tokens.size() + count); }

void InputIdentity::append_generated(std::span<const TokenId> ids) {
    prepare_generated(ids.size());
    digests.append_generated(ids);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
}

void InputIdentity::truncate(std::uint32_t frontier) {
    if (!complete_input_frontier(view(), frontier))
        throw std::invalid_argument("GLM capture intersects media");
    tokens.resize(frontier);
    digests.truncate(frontier);
    std::erase_if(media,
                  [&](const auto& m) { return m.item.token_spans.front().begin >= frontier; });
}
} // namespace ninfer::models::glm5next
