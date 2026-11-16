#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::models::context {

// A borrowed logical sequence. The last span may describe a commit that has not yet
// been adopted; callers keep every source alive until the view is consumed.
class TokenView {
public:
    TokenView() = default;

    explicit TokenView(std::span<const TokenId> input, std::span<const TokenId> generated = {},
                       std::span<const TokenId> staged = {})
        : parts_{input, generated, staged} {}

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] TokenId operator[](std::size_t index) const;
    [[nodiscard]] TokenId back() const;
    void copy(std::size_t begin, std::span<TokenId> destination) const;
    [[nodiscard]] bool equal(std::size_t begin, std::span<const TokenId> other) const;

private:
    std::array<std::span<const TokenId>, 3> parts_;
};

// RequestPlan owns the immutable input and outlives this ledger, including while the
// request is paused. Only accepted output is appended. Checkpoints never own a ledger.
class RequestLedger {
public:
    RequestLedger() = default;

    explicit RequestLedger(std::span<const TokenId> input) : input_(input) {}

    [[nodiscard]] std::size_t size() const noexcept { return input_.size() + output_.size(); }

    [[nodiscard]] std::size_t input_size() const noexcept { return input_.size(); }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] TokenId operator[](std::size_t index) const { return view()[index]; }

    [[nodiscard]] TokenId back() const { return view().back(); }

    [[nodiscard]] std::span<const TokenId> generated() const noexcept { return output_; }

    [[nodiscard]] TokenView view(std::span<const TokenId> staged = {}) const noexcept {
        return TokenView(input_, output_, staged);
    }

    void reserve_output(std::size_t count);
    void prepare_append(std::size_t count);
    void append(std::span<const TokenId> accepted);
    void copy(std::size_t begin, std::size_t end, std::vector<TokenId>& destination) const;

private:
    std::span<const TokenId> input_;
    std::vector<TokenId> output_;
};

} // namespace ninfer::models::context
