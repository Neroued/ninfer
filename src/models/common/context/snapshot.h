#pragma once

#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

// A complete immutable recovery point. Metadata and identity are native types; the common
// owner only retains the exact StateImage and the supplied logical KV prefixes.
template <class Metadata, class Identity>
class ContextSnapshot {
public:
    ContextSnapshot(StateDirectory& states, Continuation<Metadata> continuation,
                    std::vector<HistoryPrefix> prefixes, std::shared_ptr<const Identity> identity)
        : states_(&states), continuation_(std::move(continuation)), prefixes_(std::move(prefixes)),
          identity_(std::move(identity)) {
        if (!identity_) throw std::invalid_argument("snapshot has no input identity");
        for (const auto& prefix : prefixes_)
            if (!prefix.valid())
                throw std::invalid_argument("snapshot has an empty prefix reference");
        states_->retain_checkpoint_reference(continuation_.image);
    }

    ~ContextSnapshot() { release(); }

    ContextSnapshot(const ContextSnapshot& other)
        : states_(other.states_), continuation_(other.continuation_), prefixes_(other.prefixes_),
          identity_(other.identity_) {
        if (states_) states_->retain_checkpoint_reference(continuation_.image);
    }

    ContextSnapshot(ContextSnapshot&& other) noexcept
        : states_(std::exchange(other.states_, nullptr)),
          continuation_(std::move(other.continuation_)), prefixes_(std::move(other.prefixes_)),
          identity_(std::move(other.identity_)) {}

    ContextSnapshot& operator=(const ContextSnapshot& other) {
        if (this != &other) {
            ContextSnapshot copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    ContextSnapshot& operator=(ContextSnapshot&& other) noexcept {
        if (this != &other) {
            release();
            states_       = std::exchange(other.states_, nullptr);
            continuation_ = std::move(other.continuation_);
            prefixes_     = std::move(other.prefixes_);
            identity_     = std::move(other.identity_);
        }
        return *this;
    }

    [[nodiscard]] const Continuation<Metadata>& continuation() const noexcept {
        return continuation_;
    }

    [[nodiscard]] const Identity& identity() const noexcept { return *identity_; }

    [[nodiscard]] const std::shared_ptr<const Identity>& identity_owner() const noexcept {
        return identity_;
    }

    [[nodiscard]] const HistoryPrefix& prefix(std::size_t index) const {
        return prefixes_.at(index);
    }

    [[nodiscard]] std::span<const HistoryPrefix> prefixes() const noexcept { return prefixes_; }

    [[nodiscard]] StateId take_mutable_image() {
        if (!states_) throw std::logic_error("snapshot image already consumed");
        states_->consume_checkpoint(continuation_.image);
        states_ = nullptr;
        return continuation_.image;
    }

private:
    void release() noexcept {
        if (states_ && !states_->release_checkpoint_owner(continuation_.image)) std::terminate();
        states_ = nullptr;
    }

    StateDirectory* states_;
    Continuation<Metadata> continuation_;
    std::vector<HistoryPrefix> prefixes_;
    std::shared_ptr<const Identity> identity_;
};

} // namespace ninfer::models::context
