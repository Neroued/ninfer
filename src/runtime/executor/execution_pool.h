#pragma once

#include "runtime/executor/contract.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

class ExecutionPool;

class ExecutionReservation {
public:
    ExecutionReservation() noexcept = default;

    ~ExecutionReservation() { reset(); }

    ExecutionReservation(ExecutionReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), frame_(other.frame_) {}

    ExecutionReservation& operator=(ExecutionReservation&& other) noexcept {
        if (this != &other) {
            reset();
            owner_ = std::exchange(other.owner_, nullptr);
            frame_ = other.frame_;
        }
        return *this;
    }

    ExecutionReservation(const ExecutionReservation&)            = delete;
    ExecutionReservation& operator=(const ExecutionReservation&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] FrameId frame() const noexcept { return frame_; }

private:
    ExecutionReservation(ExecutionPool& owner, FrameId frame) : owner_(&owner), frame_(frame) {}

    void reset() noexcept;

    ExecutionPool* owner_ = nullptr;
    FrameId frame_;
    friend class ExecutionPool;
};

// A frame's control commands consume its original reservation. Supply and resolution therefore
// remain admissible even when every ordinary submission slot is occupied.
class ExecutionPool {
public:
    explicit ExecutionPool(ExecutionCapacity capacity)
        : capacity_(capacity), compute_(capacity.compute_frames),
          context_(capacity.context_frames) {}

    ExecutionPool(const ExecutionPool&)            = delete;
    ExecutionPool& operator=(const ExecutionPool&) = delete;

    [[nodiscard]] ExecutionCapacity capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::optional<ExecutionReservation> try_reserve(FrameKind kind) {
        auto& slots = select(kind);
        for (std::uint32_t index = 0; index < slots.size(); ++index) {
            auto& slot = slots[index];
            if (slot.state != State::Free) continue;
            ++slot.generation;
            slot.state = State::Reserved;
            return ExecutionReservation(*this, {kind, index, slot.generation});
        }
        return {};
    }

    [[nodiscard]] FrameId accept(ExecutionReservation&& reservation) {
        if (reservation.owner_ != this) {
            throw std::logic_error("execution reservation belongs to another executor");
        }
        auto& slot = lookup(reservation.frame_);
        if (slot.state != State::Reserved) {
            throw std::logic_error("execution reservation is no longer pending");
        }
        slot.state         = State::Submitted;
        reservation.owner_ = nullptr;
        return reservation.frame_;
    }

    void retire(FrameId frame) {
        auto& slot = lookup(frame);
        if (slot.state != State::Submitted) {
            throw std::logic_error("execution frame retired without a submission");
        }
        slot.state = State::Free;
    }

    void retire_submitted() noexcept {
        const auto retire = [](auto& slots) {
            for (auto& slot : slots) {
                if (slot.state == State::Submitted) slot.state = State::Free;
            }
        };
        retire(compute_);
        retire(context_);
    }

    [[nodiscard]] bool submitted(FrameId frame) const noexcept {
        const auto& slots = select(frame.kind);
        return frame.index < slots.size() && slots[frame.index].generation == frame.generation &&
               slots[frame.index].state == State::Submitted;
    }

    [[nodiscard]] bool busy() const noexcept {
        const auto occupied = [](const auto& slot) { return slot.state != State::Free; };
        return std::any_of(compute_.begin(), compute_.end(), occupied) ||
               std::any_of(context_.begin(), context_.end(), occupied);
    }

private:
    enum class State : std::uint8_t { Free, Reserved, Submitted };

    struct Slot {
        std::uint64_t generation = 0;
        State state              = State::Free;
    };

    [[nodiscard]] std::vector<Slot>& select(FrameKind kind) noexcept {
        return kind == FrameKind::Compute ? compute_ : context_;
    }

    [[nodiscard]] const std::vector<Slot>& select(FrameKind kind) const noexcept {
        return kind == FrameKind::Compute ? compute_ : context_;
    }

    [[nodiscard]] Slot& lookup(FrameId frame) {
        auto& slots = select(frame.kind);
        if (frame.index >= slots.size() || slots[frame.index].generation != frame.generation) {
            throw std::logic_error("execution frame generation is stale");
        }
        return slots[frame.index];
    }

    void release(FrameId frame) noexcept {
        auto& slot = select(frame.kind)[frame.index];
        if (slot.generation == frame.generation && slot.state == State::Reserved) {
            slot.state = State::Free;
        }
    }

    ExecutionCapacity capacity_;
    std::vector<Slot> compute_, context_;
    friend class ExecutionReservation;
};

inline void ExecutionReservation::reset() noexcept {
    if (owner_) std::exchange(owner_, nullptr)->release(frame_);
}

} // namespace ninfer::runtime
