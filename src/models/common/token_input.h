#pragma once

#include "ninfer/types.h"

#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models {

// Forced output is the calculation input without its anchor, followed by the final token.
struct ControlTokens {
    std::span<const TokenId> input;
    TokenId last;

    [[nodiscard]] std::size_t size() const noexcept { return input.size(); }

    [[nodiscard]] TokenId operator[](std::size_t i) const noexcept {
        return i + 1 == size() ? last : input[i + 1];
    }

    void copy(std::span<TokenId> destination, std::size_t offset = 0) const;
};

struct TokenCount {
    TokenId token;
    std::int32_t count;
};

// Counts cover committed output only. Empty also represents an unused penalty history.
class TokenCounts {
public:
    using Sparse  = std::vector<TokenCount>;
    using Dense   = std::vector<std::int32_t>;
    using Storage = std::variant<std::monostate, Sparse, Dense>;

    TokenCounts() = default;

    explicit TokenCounts(Storage values) : values_(std::move(values)) {}

    [[nodiscard]] const Storage& storage() const noexcept { return values_; }

    [[nodiscard]] bool empty() const noexcept { return values_.index() == 0; }

    void validate(std::uint32_t vocabulary, std::uint32_t max_tokens) const;
    void restore(std::span<std::int32_t> destination) const;

    [[nodiscard]] static std::size_t encoded_capacity(std::uint32_t vocabulary) noexcept {
        return 9 + 4ULL * vocabulary;
    }
private:
    Storage values_;
};

// One scratch owner per Session; used only when installing a residency.
class TokenCountBuilder {
public:
    void prepare(std::uint32_t vocabulary);

    [[nodiscard]] static std::size_t capacity_bytes(std::uint32_t vocabulary) noexcept {
        return 8ULL * vocabulary;
    }

    [[nodiscard]] TokenCounts build(std::span<const TokenId>, std::uint32_t vocabulary);
private:
    std::vector<std::int32_t> counts_;
    std::vector<TokenId> touched_;
};

} // namespace ninfer::models
