#include "models/qwen3_5/program/structured_round.h"
#include "core/device.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
StructuredRound::StructuredRound(Tensor masks, std::uint32_t vocab, std::uint32_t width,
                                 std::uint32_t lanes)
    : masks_(masks), words_((vocab + 31) / 32), width_(width), lanes_(lanes),
      host_masks_(masks.bytes()), host_drafts_(std::max(1U, width - 1) * lanes * sizeof(TokenId)) {
    std::fill_n(static_cast<std::uint32_t*>(host_masks_.data()), masks.bytes() / 4,
                ~std::uint32_t{0});
}

const std::uint32_t* StructuredRound::device_mask(std::uint32_t lane) const {
    return static_cast<const std::uint32_t*>(masks_.data) + lane * width_ * words_;
}

void StructuredRound::fill(std::uint32_t lane, const text::GrammarState& grammar,
                           std::span<const TokenId> drafts, cudaStream_t stream) {
    auto* host = static_cast<std::uint32_t*>(host_masks_.data()) + lane * width_ * words_;
    grammar.fill_masks({host, (drafts.size() + 1) * words_}, drafts);
    CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint32_t*>(device_mask(lane)), host,
                               (drafts.size() + 1) * words_ * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, stream));
}

void StructuredRound::begin_dflash() {
    rows_  = {};
    error_ = nullptr;
}

void StructuredRound::set_dflash_row(std::uint32_t row, std::uint32_t lane, std::uint32_t extent,
                                     std::shared_ptr<text::GrammarState> grammar) {
    rows_.at(row) = {lane, extent, std::move(grammar)};
}

void CUDART_CB StructuredRound::callback(void* opaque) noexcept {
    auto& self = *static_cast<StructuredRound*>(opaque);
    try {
        for (std::size_t row = 0; row < self.rows_.size(); ++row) {
            const auto& entry = self.rows_[row];
            if (!entry.grammar) { continue; }
            auto* mask = static_cast<std::uint32_t*>(self.host_masks_.data()) +
                         entry.lane * self.width_ * self.words_;
            auto* drafts =
                static_cast<TokenId*>(self.host_drafts_.data()) + row * (self.width_ - 1);
            entry.grammar->fill_masks({mask, (entry.extent + 1) * self.words_},
                                      {drafts, entry.extent});
        }
    } catch (...) { self.error_ = std::current_exception(); }
}

void StructuredRound::enqueue_dflash(const Tensor& drafts, cudaStream_t stream) {
    // No CUDA calls or exceptions escape the host function. The transfers and host node are
    // captured together: verification cannot observe masks until the CPU matcher has filled them.
    CUDA_CHECK(cudaMemcpyAsync(host_drafts_.data(), drafts.data, drafts.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaLaunchHostFunc(stream, callback, this));
    CUDA_CHECK(cudaMemcpyAsync(masks_.data, host_masks_.data(), masks_.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void StructuredRound::check() const {
    if (error_) { std::rethrow_exception(error_); }
}
} // namespace ninfer::models::qwen3_5
