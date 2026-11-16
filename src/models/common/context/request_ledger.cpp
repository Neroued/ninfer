#include "models/common/context/request_ledger.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::context {

std::size_t TokenView::size() const noexcept {
    return parts_[0].size() + parts_[1].size() + parts_[2].size();
}

TokenId TokenView::operator[](std::size_t index) const {
    for (auto part : parts_) {
        if (index < part.size()) return part[index];
        index -= part.size();
    }
    throw std::out_of_range("token view index exceeds the request ledger");
}

TokenId TokenView::back() const {
    for (std::size_t i = parts_.size(); i; --i)
        if (!parts_[i - 1].empty()) return parts_[i - 1].back();
    throw std::out_of_range("empty token view has no last token");
}

void TokenView::copy(std::size_t begin, std::span<TokenId> destination) const {
    if (begin > size() || destination.size() > size() - begin)
        throw std::out_of_range("token range exceeds the request ledger");
    for (auto part : parts_) {
        if (begin >= part.size()) {
            begin -= part.size();
            continue;
        }
        const auto count = std::min(destination.size(), part.size() - begin);
        std::copy_n(part.begin() + begin, count, destination.begin());
        destination = destination.subspan(count);
        begin       = 0;
        if (destination.empty()) break;
    }
}

bool TokenView::equal(std::size_t begin, std::span<const TokenId> other) const {
    if (begin > size() || other.size() > size() - begin) return false;
    for (auto part : parts_) {
        if (begin >= part.size()) {
            begin -= part.size();
            continue;
        }
        const auto count = std::min(other.size(), part.size() - begin);
        if (!std::equal(other.begin(), other.begin() + count, part.begin() + begin)) return false;
        other = other.subspan(count);
        begin = 0;
        if (other.empty()) break;
    }
    return true;
}

void RequestLedger::reserve_output(std::size_t count) { output_.reserve(count); }

void RequestLedger::prepare_append(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() - output_.size())
        throw std::overflow_error("request output length overflows size_t");
    output_.reserve(output_.size() + count);
}

void RequestLedger::append(std::span<const TokenId> accepted) {
    if (accepted.size() > output_.capacity() - output_.size())
        throw std::logic_error("request output was not reserved before commit");
    output_.insert(output_.end(), accepted.begin(), accepted.end());
}

void RequestLedger::copy(std::size_t begin, std::size_t end,
                         std::vector<TokenId>& destination) const {
    if (begin > end || end > size())
        throw std::out_of_range("token range exceeds the request ledger");
    destination.resize(end - begin);
    view().copy(begin, destination);
}

} // namespace ninfer::models::context
