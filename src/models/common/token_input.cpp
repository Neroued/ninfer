#include "models/common/token_input.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models {

void ControlTokens::copy(std::span<TokenId> destination, std::size_t offset) const {
    if (offset > size() || destination.size() > size() - offset)
        throw std::out_of_range("control output slice exceeds its input");
    for (std::size_t i = 0; i < destination.size(); ++i) destination[i] = (*this)[offset + i];
}

void TokenCounts::validate(std::uint32_t vocabulary, std::uint32_t max_tokens) const {
    std::uint64_t total = 0;
    if (const auto* sparse = std::get_if<Sparse>(&values_)) {
        if (sparse->empty() || sparse->size() * 2ULL >= vocabulary)
            throw std::invalid_argument("token counts have a noncanonical sparse encoding");
        TokenId previous = -1;
        for (auto [token, count] : *sparse) {
            if (token <= previous || std::uint32_t(token) >= vocabulary || count <= 0)
                throw std::invalid_argument("token counts have an invalid sparse entry");
            previous = token;
            total += count;
        }
    } else if (const auto* dense = std::get_if<Dense>(&values_)) {
        if (dense->size() != vocabulary)
            throw std::invalid_argument("token counts differ from vocabulary");
        for (auto count : *dense) {
            if (count < 0) throw std::invalid_argument("token count is negative");
            total += count;
        }
    }
    if (total > max_tokens) throw std::invalid_argument("token counts exceed committed capacity");
}

void TokenCounts::restore(std::span<std::int32_t> destination) const {
    if (const auto* dense = std::get_if<Dense>(&values_)) {
        if (dense->size() != destination.size())
            throw std::invalid_argument("token count destination differs from vocabulary");
        std::copy(dense->begin(), dense->end(), destination.begin());
    } else {
        std::fill(destination.begin(), destination.end(), 0);
        if (const auto* sparse = std::get_if<Sparse>(&values_))
            for (auto [token, count] : *sparse) {
                if (token < 0 || std::size_t(token) >= destination.size())
                    throw std::out_of_range("token count destination omits an entry");
                destination[token] = count;
            }
    }
}

void TokenCountBuilder::prepare(std::uint32_t vocabulary) {
    counts_.resize(vocabulary, 0);
    touched_.reserve(vocabulary);
}

TokenCounts TokenCountBuilder::build(std::span<const TokenId> tokens, std::uint32_t vocabulary) {
    if (tokens.empty()) return {};
    if (tokens.size() > std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("token counts exceed int32 capacity");
    counts_.resize(vocabulary, 0);
    touched_.reserve(std::min<std::size_t>(tokens.size(), vocabulary));
    const auto clear = [&] {
        for (auto token : touched_) counts_[token] = 0;
        touched_.clear();
    };
    try {
        for (auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= vocabulary)
                throw std::invalid_argument("generated token exceeds vocabulary");
            if (counts_[token]++ == 0) touched_.push_back(token);
        }
        TokenCounts result;
        if (touched_.size() * 2ULL < vocabulary) {
            std::sort(touched_.begin(), touched_.end());
            TokenCounts::Sparse values;
            values.reserve(touched_.size());
            for (auto token : touched_) values.push_back({token, counts_[token]});
            result = TokenCounts(std::move(values));
        } else {
            result = TokenCounts(counts_);
        }
        clear();
        return result;
    } catch (...) {
        clear();
        throw;
    }
}

} // namespace ninfer::models
