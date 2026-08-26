#pragma once

#include <ninfer/targets/qwen3_6/state_image.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class StateImageStore;

class StateImageHandle {
public:
    StateImageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(StateImageHandle, StateImageHandle) noexcept = default;

private:
    StateImageHandle(const StateImageStore* owner, std::uint32_t index,
                     std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const StateImageStore* owner_ = nullptr;
    std::uint32_t index_          = 0;
    std::uint32_t generation_     = 0;

    friend class StateImageStore;
};

enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};

enum class StateReplicaResidency : std::uint8_t {
    None,
    DeviceOnly,
    HostOnly,
    Both,
};

enum class StateTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    HostToFork,
};

struct StateImageSelectors {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

class StateImageTransfer {
public:
    StateImageTransfer() noexcept = default;
    ~StateImageTransfer();

    StateImageTransfer(StateImageTransfer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), id_(std::exchange(other.id_, 0)),
          direction_(other.direction_), source_(other.source_), destination_(other.destination_) {}

    StateImageTransfer& operator=(StateImageTransfer&&)      = delete;
    StateImageTransfer(const StateImageTransfer&)            = delete;
    StateImageTransfer& operator=(const StateImageTransfer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] StateTransferDirection direction() const noexcept { return direction_; }

private:
    StateImageTransfer(StateImageStore& owner, std::uint64_t id, StateTransferDirection direction,
                       StateImageHandle source, StateImageHandle destination) noexcept
        : owner_(&owner), id_(id), direction_(direction), source_(source),
          destination_(destination) {}

    StateImageStore* owner_           = nullptr;
    std::uint64_t id_                 = 0;
    StateTransferDirection direction_ = StateTransferDirection::DeviceToHost;
    StateImageHandle source_;
    StateImageHandle destination_;

    friend class StateImageStore;
};

// Program-private logical StateImage ownership. Logical identity is independent from Device and
// Host replicas; published raw Tensor views are reconstructed only from an active Device binding.
class StateImageStore {
public:
    StateImageStore(qwen3_6::StateImageDevicePool& device, qwen3_6::HostStatePool* host,
                    std::uint32_t logical_capacity)
        : device_(&device), host_(host), objects_(logical_capacity),
          free_objects_(logical_capacity),
          free_device_slots_(static_cast<std::size_t>(device.slot_count())),
          free_object_count_(logical_capacity),
          free_device_count_(static_cast<std::uint32_t>(device.slot_count())) {
        if (logical_capacity == 0 || device.slot_count() <= 0 ||
            logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
            (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                                       host->capacity())) {
            throw std::invalid_argument("StateImageStore capacity is inconsistent");
        }
        for (std::uint32_t index = 0; index < logical_capacity; ++index) {
            free_objects_[index] = logical_capacity - 1U - index;
        }
        for (std::uint32_t index = 0; index < free_device_slots_.size(); ++index) {
            free_device_slots_[index] =
                static_cast<std::int32_t>(free_device_slots_.size() - 1U - index);
        }
    }

    StateImageStore(const StateImageStore&)            = delete;
    StateImageStore& operator=(const StateImageStore&) = delete;
    StateImageStore(StateImageStore&&)                 = delete;
    StateImageStore& operator=(StateImageStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(objects_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return capacity() - free_object_count_;
    }

    [[nodiscard]] std::uint32_t device_occupied() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size()) - free_device_count_;
    }

    [[nodiscard]] std::uint32_t device_capacity() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size());
    }

    [[nodiscard]] std::uint32_t host_occupied() const noexcept {
        return host_ == nullptr ? 0U : host_->occupied();
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, true);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_logical_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, false);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_reset(cudaStream_t stream = nullptr) {
        std::optional<StateImageHandle> handle = allocate(StateImageRole::ActiveMutable, true);
        if (!handle) { return std::nullopt; }
        try {
            Object& object = require(*handle);
            device_->zero_slot(*object.device_slot, stream);
            object.content_epoch = next_epoch();
        } catch (...) {
            (void)release(*handle);
            throw;
        }
        return handle;
    }

    void activate_reset(StateImageHandle handle, cudaStream_t stream = nullptr) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ReservedDestination || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage reset reservation is not activatable");
        }
        device_->zero_slot(*object.device_slot, stream);
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool valid(StateImageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < objects_.size() &&
               objects_[handle.index_].role != StateImageRole::Free &&
               objects_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(StateImageHandle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] StateImageRole role(StateImageHandle handle) const {
        return require(handle).role;
    }

    [[nodiscard]] StateReplicaResidency residency(StateImageHandle handle) const {
        const Object& object = require(handle);
        if (object.device_slot && object.host_slot) { return StateReplicaResidency::Both; }
        if (object.device_slot) { return StateReplicaResidency::DeviceOnly; }
        if (object.host_slot) { return StateReplicaResidency::HostOnly; }
        return StateReplicaResidency::None;
    }

    [[nodiscard]] std::uint64_t content_epoch(StateImageHandle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t source_pins(StateImageHandle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] std::uint32_t checkpoint_references(StateImageHandle handle) const {
        return require(handle).checkpoint_references;
    }

    void retain_checkpoint_reference(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("StateImage checkpoint reference is not retainable");
        }
        ++object.checkpoint_references;
    }

    void release_checkpoint_reference(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == 0) {
            throw std::logic_error("StateImage checkpoint reference is not releasable");
        }
        --object.checkpoint_references;
    }

    [[nodiscard]] std::int32_t physical_slot(StateImageHandle handle) const {
        const Object& object = require(handle);
        if (!object.device_slot) {
            throw std::logic_error("StateImage has no published Device replica");
        }
        return *object.device_slot;
    }

    void move_checkpoint_to_active(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not movable");
        }
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) {
                throw std::logic_error("StateImage Host replica could not be consumed by Move");
            }
            object.host_slot.reset();
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool drop_device_replica(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            !object.host_slot || object.source_pins != 0 || object.destination_pinned ||
            has_pending_replica(object)) {
            return false;
        }
        return_device_slot(*object.device_slot);
        object.device_slot.reset();
        return true;
    }

    [[nodiscard]] bool drop_host_replica(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || !object.host_slot || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object) ||
            !host_->release(*object.host_slot)) {
            return false;
        }
        object.host_slot.reset();
        return true;
    }

    // Transfers the Device replica to a new logical identity while the old immutable identity
    // keeps its Host replica. This is the zero-copy preserving branch for a Both source.
    void split_device_replica_identity(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            !source_object.device_slot || !source_object.host_slot ||
            source_object.source_pins != 0 || source_object.destination_pinned ||
            has_pending_replica(source_object) ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.device_slot || destination_object.host_slot ||
            has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
            destination_object.destination_pinned) {
            throw std::logic_error("StateImage replica identity split is invalid");
        }
        destination_object.device_slot   = source_object.device_slot;
        destination_object.content_epoch = source_object.content_epoch;
        destination_object.role          = StateImageRole::ActiveMutable;
        source_object.device_slot.reset();
    }

    void freeze(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ActiveMutable || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage active image is not freezable");
        }
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::CheckpointImmutable;
    }

    void thaw(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not thawable");
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool can_recycle_checkpoint_destination(StateImageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Object& object = objects_[handle.index_];
        return object.role == StateImageRole::CheckpointImmutable &&
               object.device_slot.has_value() && !object.host_slot &&
               object.checkpoint_references == 1 && object.source_pins == 0 &&
               !object.destination_pinned && !has_pending_replica(object);
    }

    [[nodiscard]] std::uint64_t recycle_checkpoint_destination(StateImageHandle handle) {
        if (!can_recycle_checkpoint_destination(handle)) {
            throw std::logic_error("StateImage checkpoint is not recyclable as a destination");
        }
        Object& object               = require(handle);
        const std::uint64_t epoch    = object.content_epoch;
        object.checkpoint_references = 0;
        object.role                  = StateImageRole::ReservedDestination;
        return epoch;
    }

    void restore_recycled_checkpoint(StateImageHandle handle, std::uint64_t content_epoch) {
        Object& object = require(handle);
        if (content_epoch == 0 || object.role != StateImageRole::ReservedDestination ||
            !object.device_slot || object.host_slot || object.checkpoint_references != 0 ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage recycled checkpoint is not restorable");
        }
        object.content_epoch         = content_epoch;
        object.checkpoint_references = 1;
        object.role                  = StateImageRole::CheckpointImmutable;
    }

    [[nodiscard]] StateImageSelectors begin_fork(StateImageHandle source,
                                                 StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            !source_object.device_slot ||
            source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
            destination_object.role != StateImageRole::ReservedDestination ||
            !destination_object.device_slot || destination_object.source_pins != 0 ||
            destination_object.destination_pinned || has_pending_replica(destination_object)) {
            throw std::logic_error("StateImage fork binding is invalid");
        }
        ++source_object.source_pins;
        destination_object.destination_pinned = true;
        destination_object.role               = StateImageRole::ActiveMutable;
        destination_object.content_epoch      = source_object.content_epoch;
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    void commit_fork(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.device_slot || !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork commit is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
    }

    void abort_fork(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.device_slot || !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork abort is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
        destination_object.role               = StateImageRole::ReservedDestination;
        destination_object.content_epoch      = 0;
    }

    [[nodiscard]] StateImageSelectors selectors(StateImageHandle source,
                                                StateImageHandle destination) const {
        const Object& source_object      = require(source);
        const Object& destination_object = require(destination);
        const bool inplace               = source == destination;
        if (!source_object.device_slot || !destination_object.device_slot ||
            (inplace && (source_object.role != StateImageRole::ActiveMutable ||
                         source_object.source_pins != 0 || source_object.destination_pinned)) ||
            (!inplace && (source_object.role != StateImageRole::CheckpointImmutable ||
                          source_object.source_pins == 0 ||
                          destination_object.role != StateImageRole::ActiveMutable ||
                          !destination_object.destination_pinned))) {
            throw std::logic_error("StateImage execution binding is invalid");
        }
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    reserve_device_to_host(StateImageHandle source) {
        Object& object = require(source);
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || object.host_slot || has_pending_replica(object) ||
            object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        std::optional<qwen3_6::HostStateSlotHandle> target = host_->allocate();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer = next_transfer();
        object.pending_host_slot     = *target;
        object.transfer_id           = transfer;
        object.destination_pinned    = true;
        ++object.source_pins;
        return StateImageTransfer(*this, transfer, StateTransferDirection::DeviceToHost, source,
                                  source);
    }

    void enqueue_device_to_host(const StateImageTransfer& transfer, cudaStream_t stream = nullptr) {
        validate_transfer(transfer);
        Object& object = require(transfer.source_);
        if (transfer.direction_ != StateTransferDirection::DeviceToHost || host_ == nullptr ||
            !object.device_slot || !object.pending_host_slot) {
            throw std::logic_error("StateImage Device-to-Host transfer is not enqueueable");
        }
        device_->copy_to_host(*object.device_slot, host_->writable_view(*object.pending_host_slot),
                              stream);
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    begin_device_to_host(StateImageHandle source, cudaStream_t stream = nullptr) {
        std::optional<StateImageTransfer> transfer = reserve_device_to_host(source);
        if (!transfer) { return std::nullopt; }
        try {
            enqueue_device_to_host(*transfer, stream);
        } catch (...) {
            abort_transfer(std::move(*transfer));
            throw;
        }
        return transfer;
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    begin_host_to_device(StateImageHandle source, cudaStream_t stream = nullptr) {
        Object& object = require(source);
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.host_slot || object.device_slot || has_pending_replica(object) ||
            object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        const std::optional<std::int32_t> target = take_device_slot();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer = next_transfer();
        object.pending_device_slot   = *target;
        object.transfer_id           = transfer;
        object.destination_pinned    = true;
        ++object.source_pins;
        try {
            device_->copy_from_host(host_->view(*object.host_slot), *target, stream);
        } catch (...) {
            --object.source_pins;
            object.destination_pinned = false;
            object.transfer_id        = 0;
            object.pending_device_slot.reset();
            return_device_slot(*target);
            throw;
        }
        return StateImageTransfer(*this, transfer, StateTransferDirection::HostToDevice, source,
                                  source);
    }

    [[nodiscard]] std::optional<StateImageTransfer> begin_host_fork(StateImageHandle source,
                                                                    StateImageHandle destination,
                                                                    cudaStream_t stream = nullptr) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (host_ == nullptr || source == destination ||
            source_object.role != StateImageRole::CheckpointImmutable || !source_object.host_slot ||
            source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.device_slot || destination_object.host_slot ||
            has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
            destination_object.destination_pinned) {
            return std::nullopt;
        }
        const std::optional<std::int32_t> target = take_device_slot();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer           = next_transfer();
        destination_object.pending_device_slot = *target;
        destination_object.transfer_id         = transfer;
        destination_object.destination_pinned  = true;
        destination_object.role                = StateImageRole::ActiveMutable;
        destination_object.content_epoch       = source_object.content_epoch;
        ++source_object.source_pins;
        try {
            device_->copy_from_host(host_->view(*source_object.host_slot), *target, stream);
        } catch (...) {
            --source_object.source_pins;
            destination_object.pending_device_slot.reset();
            destination_object.transfer_id        = 0;
            destination_object.destination_pinned = false;
            destination_object.role               = StateImageRole::ReservedDestination;
            destination_object.content_epoch      = 0;
            return_device_slot(*target);
            throw;
        }
        return StateImageTransfer(*this, transfer, StateTransferDirection::HostToFork, source,
                                  destination);
    }

    void publish_transfer(StateImageTransfer&& transfer, bool keep_source_replica) {
        validate_transfer(transfer);
        Object& source      = require(transfer.source_);
        Object& destination = require(transfer.destination_);
        switch (transfer.direction_) {
        case StateTransferDirection::DeviceToHost:
            source.host_slot = source.pending_host_slot;
            source.pending_host_slot.reset();
            if (!keep_source_replica) {
                return_device_slot(*source.device_slot);
                source.device_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToDevice:
            source.device_slot = source.pending_device_slot;
            source.pending_device_slot.reset();
            if (!keep_source_replica) {
                if (host_ == nullptr || !host_->release(*source.host_slot)) {
                    throw std::logic_error("StateImage Host replica release failed at publication");
                }
                source.host_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToFork:
            destination.device_slot = destination.pending_device_slot;
            destination.pending_device_slot.reset();
            --source.source_pins;
            destination.destination_pinned = false;
            destination.transfer_id        = 0;
            break;
        }
        consume(transfer);
    }

    void abort_transfer(StateImageTransfer&& transfer) noexcept {
        if (!valid_transfer(transfer)) {
            consume(transfer);
            return;
        }
        Object& source      = objects_[transfer.source_.index_];
        Object& destination = objects_[transfer.destination_.index_];
        if (source.source_pins != 0) { --source.source_pins; }
        if (transfer.direction_ == StateTransferDirection::DeviceToHost) {
            if (host_ != nullptr && source.pending_host_slot) {
                (void)host_->release(*source.pending_host_slot);
            }
            source.pending_host_slot.reset();
            source.destination_pinned = false;
            source.transfer_id        = 0;
        } else {
            if (destination.pending_device_slot) {
                return_device_slot(*destination.pending_device_slot);
            }
            destination.pending_device_slot.reset();
            destination.destination_pinned = false;
            destination.transfer_id        = 0;
            if (transfer.direction_ == StateTransferDirection::HostToFork) {
                destination.role          = StateImageRole::ReservedDestination;
                destination.content_epoch = 0;
            }
        }
        consume(transfer);
    }

    [[nodiscard]] bool release(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            return false;
        }
        if (object.device_slot) {
            return_device_slot(*object.device_slot);
            object.device_slot.reset();
        }
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) { return false; }
            object.host_slot.reset();
        }
        object.role          = StateImageRole::Free;
        object.content_epoch = 0;
        if (++object.generation == 0) { ++object.generation; }
        free_objects_[free_object_count_++] = handle.index_;
        return true;
    }

private:
    struct Object {
        std::uint32_t generation    = 1;
        std::uint64_t content_epoch = 0;
        std::optional<std::int32_t> device_slot;
        std::optional<qwen3_6::HostStateSlotHandle> host_slot;
        std::optional<std::int32_t> pending_device_slot;
        std::optional<qwen3_6::HostStateSlotHandle> pending_host_slot;
        std::uint64_t transfer_id           = 0;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t source_pins           = 0;
        bool destination_pinned             = false;
        StateImageRole role                 = StateImageRole::Free;
    };

    [[nodiscard]] static bool has_pending_replica(const Object& object) noexcept {
        return object.pending_device_slot.has_value() || object.pending_host_slot.has_value() ||
               object.transfer_id != 0;
    }

    [[nodiscard]] std::optional<StateImageHandle> allocate(StateImageRole role,
                                                           bool with_device) noexcept {
        if (free_object_count_ == 0 || role == StateImageRole::Free ||
            (with_device && free_device_count_ == 0)) {
            return std::nullopt;
        }
        const std::uint32_t index = free_objects_[--free_object_count_];
        Object& object            = objects_[index];
        object                    = Object{.generation = object.generation, .role = role};
        if (with_device) { object.device_slot = free_device_slots_[--free_device_count_]; }
        return StateImageHandle(this, index, object.generation);
    }

    [[nodiscard]] std::optional<std::int32_t> take_device_slot() noexcept {
        if (free_device_count_ == 0) { return std::nullopt; }
        return free_device_slots_[--free_device_count_];
    }

    void return_device_slot(std::int32_t slot) noexcept {
        free_device_slots_[free_device_count_++] = slot;
    }

    [[nodiscard]] std::uint64_t next_epoch() noexcept {
        if (++next_content_epoch_ == 0) { ++next_content_epoch_; }
        return next_content_epoch_;
    }

    [[nodiscard]] std::uint64_t next_transfer() noexcept {
        if (++next_transfer_id_ == 0) { ++next_transfer_id_; }
        return next_transfer_id_;
    }

    [[nodiscard]] Object& require(StateImageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] const Object& require(StateImageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] bool valid_transfer(const StateImageTransfer& transfer) const noexcept {
        if (transfer.owner_ != this || transfer.id_ == 0 || !valid(transfer.source_) ||
            !valid(transfer.destination_)) {
            return false;
        }
        const Object& destination = objects_[transfer.destination_.index_];
        return destination.transfer_id == transfer.id_;
    }

    void validate_transfer(const StateImageTransfer& transfer) const {
        if (!valid_transfer(transfer)) {
            throw std::logic_error("StateImage transfer capability is stale");
        }
    }

    static void consume(StateImageTransfer& transfer) noexcept {
        transfer.owner_ = nullptr;
        transfer.id_    = 0;
    }

    qwen3_6::StateImageDevicePool* device_ = nullptr;
    qwen3_6::HostStatePool* host_          = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::vector<std::int32_t> free_device_slots_;
    std::uint32_t free_object_count_  = 0;
    std::uint32_t free_device_count_  = 0;
    std::uint64_t next_content_epoch_ = 0;
    std::uint64_t next_transfer_id_   = 0;
};

inline StateImageTransfer::~StateImageTransfer() {
    if (owner_ != nullptr) { owner_->abort_transfer(std::move(*this)); }
}

struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
    bool fork_pending = false;
};

} // namespace ninfer::targets::qwen3_6::detail
