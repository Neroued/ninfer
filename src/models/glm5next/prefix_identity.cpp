#include "models/glm5next/prefix_identity.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::models::glm5next {
namespace {

void validate(InputPrefixView input) {
    std::size_t end = 0;
    for (const auto& media : input.media) {
        if (media.item.token_spans.empty())
            throw std::invalid_argument("media identity has no token spans");
        for (const auto span : media.item.token_spans) {
            if (!span.count || span.begin < end || span.begin > input.tokens.size() ||
                span.count > input.tokens.size() - span.begin)
                throw std::invalid_argument("media identity spans are outside the ordered input");
            end = span.begin + span.count;
        }
        for (const auto timestamp : media.item.timestamps)
            if (!std::isfinite(timestamp))
                throw std::invalid_argument("media identity timestamps must be finite");
    }
}

} // namespace

bool complete_input_frontier(InputPrefixView input, std::size_t frontier) {
    validate(input);
    if (frontier > input.tokens.size()) return false;
    for (const auto& media : input.media) {
        const auto& spans = media.item.token_spans;
        if (spans.front().begin < frontier && frontier < spans.back().begin + spans.back().count)
            return false;
    }
    return true;
}

} // namespace ninfer::models::glm5next
