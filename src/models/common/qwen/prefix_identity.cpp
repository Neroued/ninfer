#include "models/common/qwen/prefix_identity.h"
#include <algorithm>
#include <bit>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen {

bool recovery_prefix_matches(const RecoveryIdentityView& earlier,
                             const RecoveryIdentityView& later) {
    const auto frontier = earlier.frontier;
    if (frontier > later.frontier || earlier.domain != later.domain ||
        earlier.digests.at(frontier) != later.digests.at(frontier))
        return false;
    if (earlier.backing && earlier.backing == later.backing) return true;
    return std::equal(earlier.tokens.begin(), earlier.tokens.begin() + frontier,
                      later.tokens.begin()) &&
           earlier.positions.prefix_equals(later.positions, frontier);
}

namespace {

void validate_generated(std::size_t begin, std::size_t count, std::int32_t rope_delta) {
    constexpr auto maximum = std::size_t(std::numeric_limits<std::int32_t>::max());
    if (begin > maximum || count > maximum + 1 - begin)
        throw std::overflow_error("generated prefix exceeds int32 positions");
    if (count) {
        const auto first = static_cast<std::int64_t>(begin) + rope_delta;
        const auto last  = first + static_cast<std::int64_t>(count) - 1;
        if (first < std::numeric_limits<std::int32_t>::min() ||
            last > std::numeric_limits<std::int32_t>::max())
            throw std::overflow_error("generated MRoPE position exceeds int32");
    }
}

bool same_grid(const models::frontend::VisionGrid& left,
               const models::frontend::VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<models::frontend::TokenSpan>& left,
                const std::vector<models::frontend::TokenSpan>& right) {
    return left.size() == right.size() &&
           std::equal(
               left.begin(), left.end(), right.begin(),
               [](const models::frontend::TokenSpan& a, const models::frontend::TokenSpan& b) {
                   return a.begin == b.begin && a.count == b.count;
               });
}

bool same_item(const models::frontend::VisionItem& left,
               const models::frontend::VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<models::frontend::VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const models::frontend::VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const models::frontend::TokenSpan& first = item.token_spans.front();
        const models::frontend::TokenSpan& last  = item.token_spans.back();
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

using DigestPair = std::array<std::uint64_t, 2>;

constexpr DigestPair kDigestOffset{1469598103934665603ULL, 7809847782465536322ULL};
constexpr DigestPair kDigestPrime{1099511628211ULL, 14029467366897019727ULL};
constexpr std::uint64_t kTokenDigestDomain   = 0x6e696e6665722d74ULL;
constexpr std::uint64_t kVisionDigestDomain  = 0x6e696e6665722d76ULL;

void mix_digest(DigestPair& digest, std::uint64_t value) noexcept {
    for (std::size_t lane = 0; lane < digest.size(); ++lane) {
        const std::uint64_t lane_value =
            lane == 0 ? value : std::rotl(value ^ 0x9e3779b97f4a7c15ULL, 29);
        for (std::uint32_t byte = 0; byte < 8; ++byte) {
            digest[lane] ^= static_cast<std::uint8_t>(lane_value >> (8U * byte));
            digest[lane] *= kDigestPrime[lane];
        }
    }
}

void mix_vision_item(DigestPair& digest, const models::frontend::VisionItem& item) noexcept {
    mix_digest(digest, kVisionDigestDomain);
    mix_digest(digest, static_cast<std::uint8_t>(item.modality));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.temporal));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.height));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.width));
    mix_digest(digest, item.patch_begin);
    mix_digest(digest, item.patch_count);
    for (const std::uint8_t byte : item.content_digest) { mix_digest(digest, byte); }
    mix_digest(digest, item.timestamps.size());
    for (const double timestamp : item.timestamps) {
        mix_digest(digest, std::bit_cast<std::uint64_t>(timestamp));
    }
    mix_digest(digest, item.token_spans.size());
    for (const models::frontend::TokenSpan& span : item.token_spans) {
        mix_digest(digest, span.begin);
        mix_digest(digest, span.count);
    }
}

void append_digest(std::vector<DigestPair>& digests, TokenId token, std::uint8_t token_type,
                   const std::array<std::int32_t, 3>& positions) {
    DigestPair digest = digests.back();
    mix_digest(digest, kTokenDigestDomain);
    mix_digest(digest, static_cast<std::uint32_t>(token));
    mix_digest(digest, token_type);
    for (const std::int32_t position : positions) {
        mix_digest(digest, static_cast<std::uint32_t>(position));
    }
    for (std::uint64_t& lane : digest) {
        if (lane == 0) { lane = 1; }
    }
    digests.push_back(digest);
}

std::size_t checked_vision_end(const models::frontend::VisionItem& item,
                               std::size_t prompt_tokens) {
    if (item.token_spans.empty()) {
        throw std::invalid_argument("Vision shortlist item has no token spans");
    }
    std::size_t previous_end = 0;
    for (const models::frontend::TokenSpan& span : item.token_spans) {
        if (span.count == 0 || span.begin > prompt_tokens ||
            span.count > prompt_tokens - span.begin || span.begin < previous_end) {
            throw std::invalid_argument("Vision shortlist item has invalid token spans");
        }
        previous_end = span.begin + span.count;
    }
    return previous_end;
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
    regular_prefix_ = 0;
    position_origins_.fill(0);
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    regular_prefix_ = 0;
    position_origins_.fill(0);
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        position_origins_[axis] = tokens ? positions_[axis].front() : 0;
    }
    while (regular_prefix_ < tokens && token_types_[regular_prefix_] == 0) {
        bool regular = true;
        for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
            regular = regular && static_cast<std::int64_t>(positions_[axis][regular_prefix_]) ==
                                     static_cast<std::int64_t>(position_origins_[axis]) +
                                         static_cast<std::int64_t>(regular_prefix_);
        }
        if (!regular) { break; }
        ++regular_prefix_;
    }
}

void ResidentPrefixIdentity::swap(ResidentPrefixIdentity& other) noexcept {
    token_types_.swap(other.token_types_);
    positions_.swap(other.positions_);
    vision_items_.swap(other.vision_items_);
    std::swap(regular_prefix_, other.regular_prefix_);
    position_origins_.swap(other.position_origins_);
}

void ResidentPrefixIdentity::prepare_generated(std::size_t count, std::int32_t rope_delta) {
    validate_generated(size(), count, rope_delta);
    reserve(size() + count);
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
        if (index == 0) {
            regular_prefix_ = 0;
            position_origins_.fill(rope_delta);
        }
        if (regular_prefix_ == index &&
            std::all_of(position_origins_.begin(), position_origins_.end(),
                        [&](auto origin) { return origin == rope_delta; })) {
            ++regular_prefix_;
        }
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
    regular_prefix_ = std::min(regular_prefix_, tokens);
    if (!tokens) { position_origins_.fill(0); }
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
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
    return true;
}

bool ResidentPrefixIdentity::equals(const ResidentPrefixIdentity& other) const {
    return size() == other.size() && prefix_equals(other, size());
}

bool ResidentPrefixIdentity::prefix_equals(const ResidentPrefixIdentity& other,
                                           std::size_t count) const {
    if (count > size() || count > other.size()) { return false; }
    const bool regular = count <= regular_prefix_ && count <= other.regular_prefix_ &&
                         (!count || position_origins_ == other.position_origins_);
    if (!regular) {
        if (!std::equal(token_types_.begin(),
                        token_types_.begin() + static_cast<std::ptrdiff_t>(count),
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
    return true;
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
    digests_.clear();
    reserve(tokens);
    digests_.push_back(kDigestOffset);
    std::size_t next_vision  = 0;
    std::size_t next_vision_end =
        prompt.vision_items.empty() ? 0 : checked_vision_end(prompt.vision_items.front(), tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        const std::array<std::int32_t, 3> positions{prompt.positions[index],
                                                    prompt.positions[tokens + index],
                                                    prompt.positions[2U * tokens + index]};
        append_digest(digests_, prompt.token_ids[index], prompt.token_types[index], positions);
        const std::size_t frontier = index + 1U;
        while (next_vision < prompt.vision_items.size() && next_vision_end == frontier) {
            mix_vision_item(digests_.back(), prompt.vision_items[next_vision]);
            ++next_vision;
            if (next_vision < prompt.vision_items.size()) {
                next_vision_end = checked_vision_end(prompt.vision_items[next_vision], tokens);
                if (next_vision_end < frontier) {
                    throw std::invalid_argument("Vision shortlist items are not prefix ordered");
                }
            }
        }
    }
    if (next_vision != prompt.vision_items.size()) {
        throw std::invalid_argument("Vision shortlist item exceeds the prompt");
    }
}

void PrefixShortlistDigests::swap(PrefixShortlistDigests& other) noexcept {
    digests_.swap(other.digests_);
}

void PrefixShortlistDigests::prepare_generated(std::size_t count, std::int32_t rope_delta) {
    if (digests_.empty())
        throw std::logic_error("generated shortlist append has no resident prefix");
    validate_generated(size(), count, rope_delta);
    reserve(size() + count);
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
        append_digest(digests_, tokens[offset], 0, {value, value, value});
    }
}

void PrefixShortlistDigests::truncate(std::size_t tokens) {
    if (digests_.empty() || tokens > size()) {
        throw std::out_of_range("cannot extend prefix shortlist by truncation");
    }
    digests_.resize(tokens + 1U);
}

std::array<std::uint64_t, 2> PrefixShortlistDigests::at(std::size_t frontier) const {
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

} // namespace ninfer::models::qwen
