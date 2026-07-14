#pragma once

#include "runtime/contract/types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

// PendingRound is a transaction handle over token storage owned by Owner. Owner
// must outlive the handle and must reject callbacks whose epoch is no longer the
// currently pending round. The discard callback must never affect a newer epoch.
template <class Owner>
class PendingRound {
public:
    struct Callbacks {
        bool (*is_live)(const Owner&, std::uint64_t round_epoch) noexcept = nullptr;
        void (*commit_all)(Owner&, std::uint64_t round_epoch)             = nullptr;
        FinishDisposition (*commit_prefix_and_finish)(Owner&, std::uint64_t round_epoch,
                                                      std::size_t count)  = nullptr;
        void (*discard)(Owner&, std::uint64_t round_epoch) noexcept       = nullptr;
    };

    PendingRound(Owner& owner, std::uint64_t round_epoch, std::span<const TokenId> tokens,
                 Callbacks callbacks)
        : owner_(&owner), round_epoch_(round_epoch), tokens_(tokens), callbacks_(callbacks) {
        if (tokens_.empty()) {
            throw std::invalid_argument("a pending round must contain at least one token");
        }
        if (callbacks_.is_live == nullptr || callbacks_.commit_all == nullptr ||
            callbacks_.commit_prefix_and_finish == nullptr || callbacks_.discard == nullptr) {
            throw std::invalid_argument("a pending round requires all owner callbacks");
        }
    }

    ~PendingRound() noexcept { discard_if_live(); }

    PendingRound(const PendingRound&)            = delete;
    PendingRound& operator=(const PendingRound&) = delete;

    PendingRound(PendingRound&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          round_epoch_(std::exchange(other.round_epoch_, 0)), tokens_(other.tokens_),
          callbacks_(other.callbacks_) {
        other.tokens_    = {};
        other.callbacks_ = {};
    }

    PendingRound& operator=(PendingRound&&) = delete;

    [[nodiscard]] std::span<const TokenId> tokens() const {
        if (owner_ == nullptr) { return {}; }
        require_live();
        return tokens_;
    }

    void commit_all() && {
        require_live();
        callbacks_.commit_all(*owner_, round_epoch_);
        disarm();
    }

    [[nodiscard]] FinishDisposition commit_prefix_and_finish(std::size_t count) && {
        require_live();
        if (count == 0 || count > tokens_.size()) {
            throw std::out_of_range("committed prefix must be within the pending round");
        }

        const FinishDisposition disposition =
            callbacks_.commit_prefix_and_finish(*owner_, round_epoch_, count);
        disarm();
        return disposition;
    }

    void discard() && noexcept { discard_if_live(); }

private:
    void require_live() const {
        if (owner_ == nullptr) { throw std::logic_error("pending round is already resolved"); }
        if (!callbacks_.is_live(*owner_, round_epoch_)) {
            throw std::logic_error("pending round epoch is stale");
        }
    }

    void discard_if_live() noexcept {
        if (owner_ == nullptr) { return; }
        callbacks_.discard(*owner_, round_epoch_);
        disarm();
    }

    void disarm() noexcept {
        owner_       = nullptr;
        round_epoch_ = 0;
        tokens_      = {};
        callbacks_   = {};
    }

    Owner* owner_              = nullptr;
    std::uint64_t round_epoch_ = 0;
    std::span<const TokenId> tokens_;
    Callbacks callbacks_;
};

template <class Owner>
struct BeginRound {
    BeginSummary summary;
    PendingRound<Owner> round;

    BeginRound(BeginSummary begin_summary, PendingRound<Owner>&& pending_round) noexcept
        : summary(begin_summary), round(std::move(pending_round)) {}

    BeginRound(const BeginRound&)            = delete;
    BeginRound& operator=(const BeginRound&) = delete;
    BeginRound(BeginRound&&) noexcept        = default;
    BeginRound& operator=(BeginRound&&)      = delete;
};

} // namespace ninfer::runtime
