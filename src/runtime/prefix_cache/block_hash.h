#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime::prefix_cache {

// Hybrid prefix cache block geometry. It equals the paged KV page size so one tree node owns
// exactly one page group per enabled pool.
inline constexpr std::uint32_t kBlockTokens = 64;

// Lookup hash of the empty prefix. Block b's hash chains the hash of block b-1.
inline constexpr std::uint64_t kRootLookupHash = 0x6e696e6665722d68ULL;

// Lookup hash of one full block. The hash only selects a child; equality is always decided by an
// exact token and extra-key comparison, so collisions cost a comparison, never a false hit.
[[nodiscard]] std::uint64_t block_lookup_hash(std::uint64_t parent_hash,
                                              std::span<const TokenId> block_tokens,
                                              std::uint64_t extra) noexcept;

// Chained lookup hashes of every full block of `tokens`. `extras` is empty (all zero) or holds one
// extra key per full block.
[[nodiscard]] std::vector<std::uint64_t> block_lookup_hashes(std::span<const TokenId> tokens,
                                                             std::span<const std::uint64_t> extras);

} // namespace ninfer::runtime::prefix_cache
