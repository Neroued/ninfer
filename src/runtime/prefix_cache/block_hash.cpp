#include "runtime/prefix_cache/block_hash.h"

#include <stdexcept>

namespace ninfer::runtime::prefix_cache {
namespace {

constexpr std::uint64_t kMultiplier = 0x9e3779b97f4a7c15ULL;

constexpr std::uint64_t finalize(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

constexpr std::uint64_t absorb(std::uint64_t state, std::uint64_t word) noexcept {
    state ^= word;
    state *= kMultiplier;
    return state ^ (state >> 29U);
}

} // namespace

std::uint64_t block_lookup_hash(std::uint64_t parent_hash, std::span<const TokenId> block_tokens,
                                std::uint64_t extra) noexcept {
    std::uint64_t state = absorb(finalize(parent_hash), extra);
    std::size_t index   = 0;
    for (; index + 1 < block_tokens.size(); index += 2) {
        const std::uint64_t word =
            static_cast<std::uint32_t>(block_tokens[index]) |
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(block_tokens[index + 1]))
             << 32U);
        state = absorb(state, word);
    }
    if (index < block_tokens.size()) {
        state = absorb(state, static_cast<std::uint32_t>(block_tokens[index]));
    }
    return finalize(absorb(state, block_tokens.size()));
}

std::vector<std::uint64_t> block_lookup_hashes(std::span<const TokenId> tokens,
                                               std::span<const std::uint64_t> extras) {
    const std::size_t blocks = tokens.size() / kBlockTokens;
    if (!extras.empty() && extras.size() < blocks) {
        throw std::invalid_argument("prefix cache block extras do not cover every full block");
    }
    std::vector<std::uint64_t> hashes(blocks);
    std::uint64_t parent = kRootLookupHash;
    for (std::size_t block = 0; block < blocks; ++block) {
        parent = block_lookup_hash(parent, tokens.subspan(block * kBlockTokens, kBlockTokens),
                                   extras.empty() ? 0ULL : extras[block]);
        hashes[block] = parent;
    }
    return hashes;
}

} // namespace ninfer::runtime::prefix_cache
