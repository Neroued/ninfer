#include "models/glm5next/prefix_identity.h"

#include <algorithm>
#include <bit>
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
    std::uint32_t previous = 0;
    for (const auto frontier : input.execution_frontiers) {
        if (frontier <= previous || frontier > input.tokens.size())
            throw std::invalid_argument(
                "input execution frontiers must be ordered and within input");
        previous = frontier;
    }
}

bool same_media(const MediaInputIdentity& a, const MediaInputIdentity& b) {
    const auto& x = a.item;
    const auto& y = b.item;
    return a.preparation_digest == b.preparation_digest && x.modality == y.modality &&
           x.content_digest == y.content_digest && x.grid.temporal == y.grid.temporal &&
           x.grid.height == y.grid.height && x.grid.width == y.grid.width &&
           x.patch_begin == y.patch_begin && x.patch_count == y.patch_count &&
           x.timestamps == y.timestamps && x.token_spans.size() == y.token_spans.size() &&
           std::equal(x.token_spans.begin(), x.token_spans.end(), y.token_spans.begin(),
                      [](auto left, auto right) {
                          return left.begin == right.begin && left.count == right.count;
                      });
}

using Digest = std::array<std::uint64_t, 2>;

void mix(Digest& digest, std::uint64_t value) {
    for (unsigned lane = 0; lane < 2; ++lane) {
        const auto word = lane ? std::rotl(value ^ 0x9e3779b97f4a7c15ULL, 29) : value;
        for (unsigned byte = 0; byte < 8; ++byte) {
            digest[lane] ^= static_cast<std::uint8_t>(word >> (8 * byte));
            digest[lane] *= lane ? 14029467366897019727ULL : 1099511628211ULL;
        }
    }
}

void mix_media(Digest& digest, const MediaInputIdentity& media) {
    const auto& item = media.item;
    mix(digest, 0x676c6d2d6d656469ULL);
    mix(digest, static_cast<unsigned>(item.modality));
    mix(digest, item.grid.temporal);
    mix(digest, item.grid.height);
    mix(digest, item.grid.width);
    mix(digest, item.patch_begin);
    mix(digest, item.patch_count);
    for (auto byte : item.content_digest) mix(digest, byte);
    for (auto byte : media.preparation_digest) mix(digest, byte);
    mix(digest, item.timestamps.size());
    for (auto timestamp : item.timestamps)
        mix(digest, std::bit_cast<std::uint64_t>(timestamp == 0 ? 0.0 : timestamp));
    mix(digest, item.token_spans.size());
    for (auto span : item.token_spans) {
        mix(digest, span.begin);
        mix(digest, span.count);
    }
}
} // namespace

bool input_prefix_equal(InputPrefixView left, InputPrefixView right, std::size_t frontier) {
    validate(left);
    validate(right);
    if (frontier > left.tokens.size() || frontier > right.tokens.size() ||
        !std::equal(left.tokens.begin(), left.tokens.begin() + frontier, right.tokens.begin()))
        return false;
    const auto a = std::upper_bound(left.execution_frontiers.begin(),
                                    left.execution_frontiers.end(), frontier);
    const auto b = std::upper_bound(right.execution_frontiers.begin(),
                                    right.execution_frontiers.end(), frontier);
    if (!std::equal(left.execution_frontiers.begin(), a, right.execution_frontiers.begin(), b))
        return false;
    std::size_t count = 0;
    while (count < left.media.size() &&
           left.media[count].item.token_spans.front().begin < frontier) {
        if (count == right.media.size() || !same_media(left.media[count], right.media[count]))
            return false;
        ++count;
    }
    return count == right.media.size() ||
           right.media[count].item.token_spans.front().begin >= frontier;
}

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

PrefixDigests::PrefixDigests(InputPrefixView input) {
    validate(input);
    digests_.reserve(input.tokens.size() + 1);
    Digest digest{1469598103934665603ULL, 7809847782465536322ULL};
    digests_.push_back(digest);
    std::size_t media = 0, split = 0;
    for (std::size_t token = 0; token < input.tokens.size(); ++token) {
        if (media < input.media.size() &&
            input.media[media].item.token_spans.front().begin == token)
            mix_media(digest, input.media[media++]);
        mix(digest, static_cast<std::uint32_t>(input.tokens[token]));
        if (split < input.execution_frontiers.size() &&
            input.execution_frontiers[split] == token + 1) {
            mix(digest, 0x676c6d2d73706c69ULL);
            mix(digest, input.execution_frontiers[split++]);
        }
        digests_.push_back(digest);
    }
}

std::array<std::uint64_t, 2> PrefixDigests::at(std::size_t frontier) const {
    return digests_.at(frontier);
}

} // namespace ninfer::models::glm5next
