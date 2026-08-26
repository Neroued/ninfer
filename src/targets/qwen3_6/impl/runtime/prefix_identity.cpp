#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

constexpr std::uint64_t kDigestOffset        = 1469598103934665603ULL;
constexpr std::uint64_t kDigestPrime         = 1099511628211ULL;
constexpr std::uint64_t kTokenDigestDomain   = 0x6e696e6665722d74ULL;
constexpr std::uint64_t kRewriteDigestDomain = 0x6e696e6665722d72ULL;

void mix_digest(std::uint64_t& digest, std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
        digest ^= static_cast<std::uint8_t>(value >> (8U * byte));
        digest *= kDigestPrime;
    }
}

void append_digest(std::vector<std::uint64_t>& digests, TokenId token, std::uint8_t token_type,
                   const std::array<std::int32_t, 3>& positions,
                   std::span<const std::uint32_t> rewrite_frontiers, std::size_t& next_rewrite) {
    std::uint64_t digest = digests.back();
    mix_digest(digest, kTokenDigestDomain);
    mix_digest(digest, static_cast<std::uint32_t>(token));
    mix_digest(digest, token_type);
    for (const std::int32_t position : positions) {
        mix_digest(digest, static_cast<std::uint32_t>(position));
    }
    const std::size_t frontier = digests.size();
    while (next_rewrite < rewrite_frontiers.size() && rewrite_frontiers[next_rewrite] == frontier) {
        mix_digest(digest, kRewriteDigestDomain);
        mix_digest(digest, rewrite_frontiers[next_rewrite]);
        ++next_rewrite;
    }
    digests.push_back(digest == 0 ? 1 : digest);
}

} // namespace

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
    rewrite_execution_frontiers_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_                = prompt.vision_items;
    rewrite_execution_frontiers_ = prompt.identity.rewrite_execution_frontiers;
}

void ResidentPrefixIdentity::swap(ResidentPrefixIdentity& other) noexcept {
    token_types_.swap(other.token_types_);
    positions_.swap(other.positions_);
    vision_items_.swap(other.vision_items_);
    rewrite_execution_frontiers_.swap(other.rewrite_execution_frontiers_);
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
    rewrite_execution_frontiers_.erase(std::upper_bound(rewrite_execution_frontiers_.begin(),
                                                        rewrite_execution_frontiers_.end(), tokens),
                                       rewrite_execution_frontiers_.end());
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    const auto incoming_end =
        std::upper_bound(prompt.identity.rewrite_execution_frontiers.begin(),
                         prompt.identity.rewrite_execution_frontiers.end(), count);
    const auto resident_end = std::upper_bound(rewrite_execution_frontiers_.begin(),
                                               rewrite_execution_frontiers_.end(), count);
    return std::distance(prompt.identity.rewrite_execution_frontiers.begin(), incoming_end) ==
               std::distance(rewrite_execution_frontiers_.begin(), resident_end) &&
           std::equal(prompt.identity.rewrite_execution_frontiers.begin(), incoming_end,
                      rewrite_execution_frontiers_.begin());
}

bool ResidentPrefixIdentity::equals(const ResidentPrefixIdentity& other) const {
    return size() == other.size() && prefix_equals(other, size());
}

bool ResidentPrefixIdentity::prefix_equals(const ResidentPrefixIdentity& other,
                                           std::size_t count) const {
    if (count > size() || count > other.size() ||
        !std::equal(token_types_.begin(), token_types_.begin() + static_cast<std::ptrdiff_t>(count),
                    other.token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        if (!std::equal(positions_[axis].begin(),
                        positions_[axis].begin() + static_cast<std::ptrdiff_t>(count),
                        other.positions_[axis].begin())) {
            return false;
        }
    }
    std::size_t left_items  = 0;
    std::size_t right_items = 0;
    if (!prefix_item_count(vision_items_, count, &left_items) ||
        !prefix_item_count(other.vision_items_, count, &right_items) || left_items != right_items) {
        return false;
    }
    for (std::size_t index = 0; index < left_items; ++index) {
        if (!same_item(vision_items_[index], other.vision_items_[index])) { return false; }
    }
    const auto left_end  = std::upper_bound(rewrite_execution_frontiers_.begin(),
                                            rewrite_execution_frontiers_.end(), count);
    const auto right_end = std::upper_bound(other.rewrite_execution_frontiers_.begin(),
                                            other.rewrite_execution_frontiers_.end(), count);
    return std::distance(rewrite_execution_frontiers_.begin(), left_end) ==
               std::distance(other.rewrite_execution_frontiers_.begin(), right_end) &&
           std::equal(rewrite_execution_frontiers_.begin(), left_end,
                      other.rewrite_execution_frontiers_.begin());
}

void PrefixShortlistDigests::reserve(std::size_t tokens) {
    if (tokens == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("prefix shortlist capacity overflows size_t");
    }
    digests_.reserve(tokens + 1U);
}

void PrefixShortlistDigests::clear() noexcept { digests_.clear(); }

void PrefixShortlistDigests::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3U * tokens) {
        throw std::invalid_argument("prepared prompt shortlist metadata has an invalid shape");
    }
    std::uint32_t previous_rewrite = 0;
    for (const std::uint32_t frontier : prompt.identity.rewrite_execution_frontiers) {
        if (frontier == 0 || frontier > tokens || frontier <= previous_rewrite) {
            throw std::invalid_argument(
                "rewrite execution frontiers must be ordered unique prompt positions");
        }
        previous_rewrite = frontier;
    }
    digests_.clear();
    reserve(tokens);
    digests_.push_back(kDigestOffset);
    std::size_t next_rewrite = 0;
    for (std::size_t index = 0; index < tokens; ++index) {
        const std::array<std::int32_t, 3> positions{prompt.positions[index],
                                                    prompt.positions[tokens + index],
                                                    prompt.positions[2U * tokens + index]};
        append_digest(digests_, prompt.token_ids[index], prompt.token_types[index], positions,
                      prompt.identity.rewrite_execution_frontiers, next_rewrite);
    }
    if (next_rewrite != prompt.identity.rewrite_execution_frontiers.size()) {
        throw std::invalid_argument("rewrite execution frontier exceeds the prompt");
    }
}

void PrefixShortlistDigests::swap(PrefixShortlistDigests& other) noexcept {
    digests_.swap(other.digests_);
}

void PrefixShortlistDigests::append_generated(std::span<const TokenId> tokens,
                                              std::int32_t rope_delta) {
    if (digests_.empty()) {
        throw std::logic_error("generated shortlist append has no resident prefix");
    }
    const std::size_t begin = size();
    if (tokens.size() > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated shortlist length overflows size_t");
    }
    std::size_t no_rewrite = 0;
    for (std::size_t offset = 0; offset < tokens.size(); ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated shortlist position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated shortlist MRoPE position exceeds int32");
        }
        const std::int32_t value = static_cast<std::int32_t>(position);
        append_digest(digests_, tokens[offset], 0, {value, value, value}, {}, no_rewrite);
    }
}

void PrefixShortlistDigests::truncate(std::size_t tokens) {
    if (digests_.empty() || tokens > size()) {
        throw std::out_of_range("cannot extend prefix shortlist by truncation");
    }
    digests_.resize(tokens + 1U);
}

std::uint64_t PrefixShortlistDigests::at(std::size_t frontier) const {
    if (digests_.empty() || frontier > size()) {
        throw std::out_of_range("prefix shortlist frontier exceeds resident identity");
    }
    return digests_[frontier];
}

bool prefix_matches(const PreparedPromptData& prompt, std::span<const TokenId> resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      resident_tokens.begin()) &&
           resident_identity.matches(prompt, count);
}

} // namespace ninfer::targets::qwen3_6::detail
