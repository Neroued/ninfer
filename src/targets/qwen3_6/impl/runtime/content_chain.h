#pragma once

// Chained 64-bit content keys over the prepared-prompt identity stream.
//
// key[i] certifies token ids, token types, all three MRoPE axes, and every vision item's
// content digest for tokens [0, i) — the full input identity licensed by resident KV, so a
// single key match is sufficient for causal KV reuse. Keys are emitted per full KV page
// (kPagedKVPageSize tokens); arbitrary-frontier keys are recomputed from the preceding page
// boundary on demand.

#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

namespace content_chain {

inline constexpr std::uint64_t kFormatVersion = 1;

[[nodiscard]] inline std::uint64_t mix(std::uint64_t value) noexcept {
    value ^= value >> 32;
    value *= 0xd6e8feb86659fd93ULL;
    value ^= value >> 32;
    value *= 0xd6e8feb86659fd93ULL;
    value ^= value >> 32;
    return value;
}

[[nodiscard]] inline std::uint64_t fold(std::uint64_t chain, std::uint64_t value) noexcept {
    return mix(chain ^ mix(value + 0x9e3779b97f4a7c15ULL));
}

struct TokenStream {
    std::span<const TokenId> ids;
    std::span<const std::uint8_t> types;
    std::array<std::span<const std::int32_t>, 3> positions;
    std::span<const VisionItem> vision_items;
};

[[nodiscard]] inline TokenStream stream_of(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    TokenStream out;
    out.ids   = prompt.token_ids;
    out.types = prompt.token_types;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        out.positions[axis] =
            std::span<const std::int32_t>(prompt.positions.data() + axis * tokens, tokens);
    }
    out.vision_items = prompt.vision_items;
    return out;
}

[[nodiscard]] inline TokenStream stream_of(const std::vector<TokenId>& ledger,
                                           const ResidentPrefixIdentity& identity) {
    TokenStream out;
    out.ids   = ledger;
    out.types = identity.token_types();
    for (std::size_t axis = 0; axis < 3; ++axis) { out.positions[axis] = identity.positions(axis); }
    out.vision_items = identity.vision_items();
    return out;
}

[[nodiscard]] inline std::uint64_t fold_vision_item(std::uint64_t chain, const VisionItem& item) {
    chain = fold(chain, static_cast<std::uint64_t>(item.modality));
    chain = fold(chain, (static_cast<std::uint64_t>(item.grid.temporal) << 42) ^
                            (static_cast<std::uint64_t>(item.grid.height) << 21) ^
                            static_cast<std::uint64_t>(item.grid.width));
    for (std::size_t word = 0; word < item.content_digest.size(); word += 8) {
        std::uint64_t bits = 0;
        for (std::size_t byte = 0; byte < 8; ++byte) {
            bits = (bits << 8) | item.content_digest[word + byte];
        }
        chain = fold(chain, bits);
    }
    for (const TokenSpan& span : item.token_spans) {
        chain = fold(chain, (static_cast<std::uint64_t>(span.begin) << 32) ^ span.count);
    }
    return chain;
}

struct Chain {
    std::vector<std::uint64_t> page_keys; // key after each full kPagedKVPageSize-token page
    std::uint64_t final_key = 0;          // key after `tokens`
    std::uint32_t tokens    = 0;

    // Chain key at an arbitrary frontier, recomputed from the preceding page boundary.
    [[nodiscard]] std::uint64_t key_at(const TokenStream& stream, std::uint32_t frontier) const {
        if (frontier == tokens) { return final_key; }
        const std::uint32_t page = frontier / kPagedKVPageSize;
        if (frontier % kPagedKVPageSize == 0) {
            return page == 0 ? seed_ : page_keys.at(page - 1);
        }
        std::uint64_t chain = page == 0 ? seed_ : page_keys.at(page - 1);
        std::size_t item    = 0;
        while (item < stream.vision_items.size() &&
               stream.vision_items[item].token_spans.front().begin <
                   page * static_cast<std::uint32_t>(kPagedKVPageSize)) {
            ++item;
        }
        for (std::uint32_t index = page * kPagedKVPageSize; index < frontier; ++index) {
            while (item < stream.vision_items.size() &&
                   stream.vision_items[item].token_spans.front().begin == index) {
                chain = fold_vision_item(chain, stream.vision_items[item]);
                ++item;
            }
            chain = fold(chain, (static_cast<std::uint64_t>(
                                     static_cast<std::uint32_t>(stream.ids[index]))
                                 << 8) ^
                                    stream.types[index]);
            chain = fold(chain, (static_cast<std::uint64_t>(
                                     static_cast<std::uint32_t>(stream.positions[0][index]))
                                 << 32) ^
                                    static_cast<std::uint32_t>(stream.positions[1][index]));
            chain = fold(chain, static_cast<std::uint32_t>(stream.positions[2][index]));
        }
        return chain;
    }

    std::uint64_t seed_ = 0;
};

[[nodiscard]] inline Chain build(const TokenStream& stream, std::uint32_t tokens,
                                 std::uint64_t salt) {
    Chain out;
    out.tokens = tokens;
    out.seed_  = fold(fold(salt, kFormatVersion), 0x636f6e74656e74ULL);
    out.page_keys.reserve(tokens / kPagedKVPageSize);
    std::uint64_t chain = out.seed_;
    std::size_t item    = 0;
    for (std::uint32_t index = 0; index < tokens; ++index) {
        while (item < stream.vision_items.size() &&
               stream.vision_items[item].token_spans.front().begin == index) {
            chain = fold_vision_item(chain, stream.vision_items[item]);
            ++item;
        }
        chain = fold(chain, (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                                 stream.ids[index]))
                             << 8) ^
                                stream.types[index]);
        chain = fold(chain, (static_cast<std::uint64_t>(
                                 static_cast<std::uint32_t>(stream.positions[0][index]))
                             << 32) ^
                                static_cast<std::uint32_t>(stream.positions[1][index]));
        chain = fold(chain, static_cast<std::uint32_t>(stream.positions[2][index]));
        if ((index + 1) % kPagedKVPageSize == 0) { out.page_keys.push_back(chain); }
    }
    out.final_key = chain;
    return out;
}

} // namespace content_chain

} // namespace ninfer::targets::qwen3_6::detail
