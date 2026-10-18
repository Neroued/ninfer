#include "models/common/context/directory/state.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {

StateDirectory::ReadLease::~ReadLease() { reset(); }

StateDirectory::ReadLease::ReadLease(ReadLease&& other) noexcept
    : directory_(std::exchange(other.directory_, nullptr)), image_(other.image_),
      epoch_(other.epoch_) {}

StateDirectory::ReadLease& StateDirectory::ReadLease::operator=(ReadLease&& other) noexcept {
    if (this != &other) {
        reset();
        directory_ = std::exchange(other.directory_, nullptr);
        image_     = other.image_;
        epoch_     = other.epoch_;
    }
    return *this;
}

void StateDirectory::ReadLease::reset() noexcept {
    if (!directory_) return;
    auto* store = std::exchange(directory_, nullptr);
    auto& image = store->require(image_);
    if (!image.source_pins || image.content_epoch != epoch_) std::terminate();
    --image.source_pins;
    store->retire_released_owner(image_);
}

std::int32_t StateDirectory::ReadLease::device_slot() const {
    if (!directory_ || directory_->content_epoch(image_) != epoch_)
        throw std::logic_error("StateImage reader has no current content");
    return directory_->physical_slot(image_);
}

HostExtent StateDirectory::ReadLease::host_extent() const {
    if (!directory_ || directory_->content_epoch(image_) != epoch_)
        throw std::logic_error("StateImage reader has no current content");
    return directory_->host_extent(image_);
}

StateDirectory::ReadLease StateDirectory::read(StateId handle) {
    auto& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable ||
        object.source_pins == std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("StateImage is not readable as an immutable replica");
    ++object.source_pins;
    return ReadLease(*this, handle, object.content_epoch);
}

StateDirectory::Destination::Destination(StateDirectory& store, StateId image)
    : directory_(&store), image_(image) {
    if (store.role(image) != StateRole::ReservedDestination)
        throw std::logic_error("StateImage destination is not reserved");
}

StateDirectory::Destination::~Destination() {
    if (!directory_) return;
    if (write_source_.valid()) directory_->abort_write(write_source_, image_);
    if (!directory_->release(image_)) std::terminate();
}

StateDirectory::Destination::Destination(Destination&& other) noexcept
    : directory_(std::exchange(other.directory_, nullptr)), image_(other.image_),
      write_source_(other.write_source_) {}

StateSelectors StateDirectory::Destination::begin_write(StateId source) {
    const auto selectors = directory_->begin_write(source, image_);
    write_source_        = source;
    return selectors;
}

StateId StateDirectory::Destination::release() noexcept {
    directory_ = nullptr;
    return std::exchange(image_, StateId{});
}

StateDirectory::Transfer::Transfer(StateDirectory& owner, std::uint64_t id,
                                   StateCopyDirection direction, ReadLease source,
                                   StateId destination) noexcept
    : owner_(&owner), id_(id), direction_(direction), source_(std::move(source)),
      destination_(destination) {}

StateDirectory::Transfer::~Transfer() {
    if (owner_) owner_->abort_transfer(*this);
}

StateDirectory::Transfer::Transfer(Transfer&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), id_(std::exchange(other.id_, 0)),
      direction_(other.direction_), source_(std::move(other.source_)),
      destination_(other.destination_),
      device_slot_(std::exchange(other.device_slot_, std::nullopt)),
      host_slot_(std::move(other.host_slot_)) {}

StateTransferDescription StateDirectory::Transfer::description() const {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->validate_transfer(*this);
    StateTransferDescription result{
        .direction   = direction_,
        .source      = source_.image(),
        .destination = destination_,
        .bytes       = owner_->image_bytes_,
        .epoch       = source_.epoch(),
    };
    if (direction_ == StateCopyDirection::DeviceToHost) {
        result.slots.source = source_.device_slot();
        result.host         = host_slot_->extent();
    } else {
        result.slots.destination = *device_slot_;
        result.host              = source_.host_extent();
    }
    return result;
}

TransferWork StateDirectory::Transfer::work() const {
    return {.payload_bytes = description().bytes, .copy_operations = 1};
}

void StateDirectory::Transfer::publish(bool keep_source_replica) {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->publish_transfer(*this, keep_source_replica);
}

StateDirectory::StateDirectory(StateDirectorySpec spec, HostAllocator* host)
    : image_bytes_(spec.image_bytes), host_(host), objects_(spec.logical_capacity),
      free_objects_(spec.logical_capacity), free_device_slots_(spec.device_slots),
      free_object_count_(spec.logical_capacity), free_device_count_(spec.device_slots) {
    if (spec.logical_capacity == 0 || spec.device_slots == 0 || spec.image_bytes == 0 ||
        spec.device_slots > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        spec.logical_capacity < spec.device_slots ||
        (host &&
         (spec.image_bytes % HostAllocator::alignment != 0 ||
          spec.image_bytes < host->minimum_allocation_bytes() ||
          spec.logical_capacity - spec.device_slots < host->capacity_bytes() / spec.image_bytes))) {
        throw std::invalid_argument("StateDirectory capacity or image geometry is inconsistent");
    }
    for (std::uint32_t index = 0; index < spec.logical_capacity; ++index) {
        free_objects_[index] = spec.logical_capacity - 1U - index;
    }
    for (std::uint32_t index = 0; index < spec.device_slots; ++index) {
        free_device_slots_[index] = static_cast<std::int32_t>(spec.device_slots - 1U - index);
    }
}

std::uint32_t StateDirectory::capacity() const noexcept {
    return static_cast<std::uint32_t>(objects_.size());
}

std::uint32_t StateDirectory::occupied() const noexcept { return capacity() - free_object_count_; }

std::uint32_t StateDirectory::device_occupied() const noexcept {
    return static_cast<std::uint32_t>(free_device_slots_.size()) - free_device_count_;
}

std::uint32_t StateDirectory::device_capacity() const noexcept {
    return static_cast<std::uint32_t>(free_device_slots_.size());
}

std::uint32_t StateDirectory::host_occupied() const noexcept { return host_images_; }

std::optional<StateId> StateDirectory::reserve_destination() noexcept {
    return allocate(StateRole::ReservedDestination, true);
}

std::optional<StateId> StateDirectory::reserve_logical_destination() noexcept {
    return allocate(StateRole::ReservedDestination, false);
}

void StateDirectory::complete_initialization(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::ReservedDestination || !object.device_slot ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage initialization has no exclusive reservation");
    }
    if (object.content_epoch == 0) { object.content_epoch = next_epoch(); }
    object.role = StateRole::ActiveMutable;
}

bool StateDirectory::valid(StateId handle) const noexcept {
    return handle.valid() && handle.index < objects_.size() &&
           objects_[handle.index].role != StateRole::Free &&
           objects_[handle.index].generation == handle.generation;
}

std::uint32_t StateDirectory::descriptor_index(StateId handle) const {
    (void)require(handle);
    return handle.index;
}

StateRole StateDirectory::role(StateId handle) const { return require(handle).role; }

StateResidency StateDirectory::residency(StateId handle) const {
    const Object& object = require(handle);
    if (object.device_slot && object.host_slot) { return StateResidency::Both; }
    if (object.device_slot) { return StateResidency::DeviceOnly; }
    if (object.host_slot) { return StateResidency::HostOnly; }
    return StateResidency::None;
}

bool StateDirectory::device_resident(StateId handle) const {
    return require(handle).device_slot.has_value();
}

bool StateDirectory::host_resident(StateId handle) const {
    return require(handle).host_slot.has_value();
}

std::uint64_t StateDirectory::content_epoch(StateId handle) const {
    return require(handle).content_epoch;
}

std::uint32_t StateDirectory::source_pins(StateId handle) const {
    return require(handle).source_pins;
}

std::uint32_t StateDirectory::checkpoint_references(StateId handle) const {
    return require(handle).checkpoint_references;
}

void StateDirectory::retain_checkpoint_reference(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable ||
        object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("StateImage checkpoint reference is not retainable");
    }
    ++object.checkpoint_references;
    object.release_pending = false;
}

bool StateDirectory::can_release_checkpoint_owner(StateId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Object& object = objects_[handle.index];
    return object.role == StateRole::CheckpointImmutable && object.checkpoint_references != 0;
}

bool StateDirectory::release_checkpoint_owner(StateId handle) noexcept {
    if (!can_release_checkpoint_owner(handle)) { return false; }
    Object& object = objects_[handle.index];
    if (--object.checkpoint_references == 0) {
        object.release_pending = true;
        retire_released_owner(handle);
    }
    return true;
}

std::int32_t StateDirectory::physical_slot(StateId handle) const {
    const Object& object = require(handle);
    if (!object.device_slot) { throw std::logic_error("StateImage has no Device storage"); }
    return *object.device_slot;
}

HostExtent StateDirectory::host_extent(StateId handle) const {
    const Object& object = require(handle);
    if (!object.host_slot) throw std::logic_error("StateImage has no Host storage");
    return object.host_slot->extent();
}

void StateDirectory::make_mutable(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable || !object.device_slot ||
        object.checkpoint_references != 0 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint is not movable");
    }
    release_host_slot(object.host_slot);
    object.role = StateRole::ActiveMutable;
}

void StateDirectory::consume_checkpoint(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable || !object.device_slot ||
        object.checkpoint_references != 1 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint has other owners or readers");
    }
    object.checkpoint_references = 0;
    object.release_pending       = false;
    release_host_slot(object.host_slot);
    object.role = StateRole::ActiveMutable;
}

bool StateDirectory::drop_device_replica(StateId handle) noexcept {
    if (!valid(handle)) { return false; }
    Object& object = objects_[handle.index];
    if (object.role != StateRole::CheckpointImmutable || !object.device_slot || !object.host_slot ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        return false;
    }
    return_device_slot(*object.device_slot);
    object.device_slot.reset();
    return true;
}

bool StateDirectory::drop_host_replica(StateId handle) noexcept {
    if (!valid(handle)) { return false; }
    Object& object = objects_[handle.index];
    if (host_ == nullptr || object.role != StateRole::CheckpointImmutable || !object.device_slot ||
        !object.host_slot || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        return false;
    }
    release_host_slot(object.host_slot);
    return true;
}

void StateDirectory::split_device_replica_identity(StateId source, StateId destination) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source == destination || source_object.role != StateRole::CheckpointImmutable ||
        !source_object.device_slot || !source_object.host_slot || source_object.source_pins != 0 ||
        source_object.destination_pinned || has_pending_replica(source_object) ||
        destination_object.role != StateRole::ReservedDestination ||
        destination_object.device_slot || destination_object.host_slot ||
        has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
        destination_object.destination_pinned) {
        throw std::logic_error("StateImage replica identity split is invalid");
    }
    destination_object.device_slot   = source_object.device_slot;
    destination_object.content_epoch = source_object.content_epoch;
    source_object.device_slot.reset();
}

void StateDirectory::freeze(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::ActiveMutable || !object.device_slot || object.source_pins != 0 ||
        object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage active image is not freezable");
    }
    object.role = StateRole::CheckpointImmutable;
}

StateSelectors StateDirectory::begin_write(StateId source, StateId destination) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source == destination || source_object.role != StateRole::CheckpointImmutable ||
        !source_object.device_slot ||
        source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
        destination_object.role != StateRole::ReservedDestination ||
        !destination_object.device_slot || destination_object.source_pins != 0 ||
        destination_object.destination_pinned || has_pending_replica(destination_object)) {
        throw std::logic_error("StateImage fork binding is invalid");
    }
    ++source_object.source_pins;
    destination_object.destination_pinned = true;
    destination_object.role               = StateRole::Writing;
    destination_object.write_source       = source;
    destination_object.content_epoch      = source_object.content_epoch;
    return {.source = *source_object.device_slot, .destination = *destination_object.device_slot};
}

void StateDirectory::complete_write(StateId source, StateId destination) {
    if (source == destination) {
        (void)selectors(source, destination);
        require(destination).content_epoch = next_epoch();
        return;
    }
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source_object.role != StateRole::CheckpointImmutable || source_object.source_pins == 0 ||
        destination_object.role != StateRole::Writing || !destination_object.device_slot ||
        !destination_object.destination_pinned || destination_object.write_source != source) {
        throw std::logic_error("StateImage fork commit is invalid");
    }
    --source_object.source_pins;
    destination_object.destination_pinned = false;
    destination_object.write_source       = {};
    destination_object.role               = StateRole::ActiveMutable;
    destination_object.content_epoch      = next_epoch();
    retire_released_owner(source);
}

void StateDirectory::abort_write(StateId source, StateId destination) {
    if (!can_abort_write(source, destination)) {
        throw std::logic_error("StateImage fork abort is invalid");
    }
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    --source_object.source_pins;
    destination_object.destination_pinned = false;
    destination_object.write_source       = {};
    destination_object.role               = StateRole::ReservedDestination;
    destination_object.content_epoch      = 0;
    retire_released_owner(source);
}

bool StateDirectory::can_abort_write(StateId source, StateId destination) const noexcept {
    if (!valid(source) || !valid(destination)) { return false; }
    const Object& source_object      = objects_[source.index];
    const Object& destination_object = objects_[destination.index];
    return source != destination && source_object.role == StateRole::CheckpointImmutable &&
           source_object.source_pins != 0 && destination_object.role == StateRole::Writing &&
           destination_object.device_slot.has_value() && destination_object.destination_pinned &&
           destination_object.write_source == source;
}

StateSelectors StateDirectory::selectors(StateId source, StateId destination) const {
    const Object& source_object      = require(source);
    const Object& destination_object = require(destination);
    const bool inplace               = source == destination;
    if (!source_object.device_slot || !destination_object.device_slot ||
        (inplace && (source_object.role != StateRole::ActiveMutable ||
                     source_object.source_pins != 0 || source_object.destination_pinned)) ||
        (!inplace &&
         (source_object.role != StateRole::CheckpointImmutable || source_object.source_pins == 0 ||
          destination_object.role != StateRole::Writing || !destination_object.destination_pinned ||
          destination_object.write_source != source))) {
        throw std::logic_error("StateImage execution binding is invalid");
    }
    return {.source = *source_object.device_slot, .destination = *destination_object.device_slot};
}

std::optional<StateDirectory::Transfer>
StateDirectory::reserve_device_to_host(StateId source, HostAllocation reserved) {
    Object& object = require(source);
    if (host_ == nullptr || object.role != StateRole::CheckpointImmutable || !object.device_slot ||
        object.host_slot || has_pending_replica(object) ||
        object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    if (!reserved.valid()) {
        auto target = host_->allocate(image_bytes_);
        if (!target) { return std::nullopt; }
        reserved = std::move(*target);
    }
    if (!reserved.belongs_to(*host_) || !reserved.reserved() || reserved.bytes() != image_bytes_) {
        throw std::invalid_argument("StateImage Host reservation has incompatible geometry");
    }
    const auto id = next_transfer();
    Transfer transfer(*this, id, StateCopyDirection::DeviceToHost, read(source), source);
    transfer.host_slot_.emplace(std::move(reserved));
    ++host_images_;
    object.transfer_id        = id;
    object.destination_pinned = true;
    return transfer;
}

std::optional<StateDirectory::Transfer> StateDirectory::reserve_host_to_device(StateId source) {
    return reserve_host_to_device(source, source);
}

std::optional<StateDirectory::Transfer>
StateDirectory::reserve_host_to_device(StateId source, StateId destination) {
    auto& from = require(source);
    auto& to   = require(destination);
    if (!from.host_slot || from.role != StateRole::CheckpointImmutable ||
        from.source_pins == std::numeric_limits<std::uint32_t>::max() || to.device_slot ||
        has_pending_replica(to) || to.destination_pinned ||
        (source != destination &&
         (to.role != StateRole::ReservedDestination || to.source_pins || to.host_slot)))
        return std::nullopt;
    const auto slot = take_device_slot();
    if (!slot) return std::nullopt;
    const auto id = next_transfer();
    Transfer transfer(*this, id, StateCopyDirection::HostToDevice, read(source), destination);
    transfer.device_slot_ = *slot;
    to.transfer_id        = id;
    to.destination_pinned = true;
    to.content_epoch      = from.content_epoch;
    return transfer;
}

void StateDirectory::publish_transfer(Transfer& transfer, bool keep_source_replica) {
    validate_transfer(transfer);
    const auto source_handle = transfer.source_.image();
    auto& source             = require(source_handle);
    auto& destination        = require(transfer.destination_);
    if (source.content_epoch != transfer.source_.epoch())
        throw std::logic_error("StateImage source changed during migration");
    if (!keep_source_replica && source_handle == transfer.destination_ && source.source_pins != 1)
        throw std::logic_error("StateImage source replica still has readers");
    if (transfer.direction_ == StateCopyDirection::DeviceToHost) {
        transfer.host_slot_->publish();
        source.host_slot.emplace(std::move(*transfer.host_slot_));
        transfer.host_slot_.reset();
        if (!keep_source_replica) {
            return_device_slot(*source.device_slot);
            source.device_slot.reset();
        }
    } else {
        destination.device_slot = std::exchange(transfer.device_slot_, std::nullopt);
        if (!keep_source_replica && source_handle == transfer.destination_)
            release_host_slot(source.host_slot);
    }
    destination.destination_pinned = false;
    destination.transfer_id        = 0;
    consume(transfer);
    transfer.source_.reset();
}

void StateDirectory::abort_transfer(Transfer& transfer) noexcept {
    if (!valid_transfer(transfer)) std::terminate();
    auto& destination = objects_[transfer.destination_.index];
    release_host_slot(transfer.host_slot_);
    if (transfer.device_slot_) {
        return_device_slot(*transfer.device_slot_);
        transfer.device_slot_.reset();
    }
    destination.destination_pinned = false;
    destination.transfer_id        = 0;
    if (transfer.source_.image() != transfer.destination_) destination.content_epoch = 0;
    consume(transfer);
    transfer.source_.reset();
}

bool StateDirectory::release(StateId handle) noexcept {
    if (!can_release(handle)) { return false; }
    Object& object = objects_[handle.index];
    release_host_slot(object.host_slot);
    if (object.device_slot) {
        return_device_slot(*object.device_slot);
        object.device_slot.reset();
    }
    object.role          = StateRole::Free;
    object.content_epoch = 0;
    if (++object.generation == 0) { ++object.generation; }
    free_objects_[free_object_count_++] = handle.index;
    return true;
}

bool StateDirectory::can_release(StateId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Object& object = objects_[handle.index];
    return object.checkpoint_references == 0 && object.source_pins == 0 &&
           !object.destination_pinned && !has_pending_replica(object);
}

bool StateDirectory::can_release_after_checkpoint_references(
    StateId handle, std::uint32_t released_references) const {
    const Object& object = require(handle);
    if (released_references > object.checkpoint_references) { return false; }
    if (object.checkpoint_references != released_references) { return true; }
    if (object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        return false;
    }
    return true;
}

void StateDirectory::retire_released_owner(StateId handle) noexcept {
    if (valid(handle) && objects_[handle.index].release_pending && can_release(handle) &&
        !release(handle)) {
        std::terminate();
    }
}

bool StateDirectory::has_pending_replica(const Object& object) noexcept {
    return object.transfer_id != 0;
}

std::optional<StateId> StateDirectory::allocate(StateRole role, bool with_device) noexcept {
    if (free_object_count_ == 0 || role == StateRole::Free ||
        (with_device && free_device_count_ == 0)) {
        return std::nullopt;
    }
    const std::uint32_t index = free_objects_[--free_object_count_];
    Object& object            = objects_[index];
    object                    = Object{.generation = object.generation, .role = role};
    if (with_device) { object.device_slot = free_device_slots_[--free_device_count_]; }
    return StateId{.index = index, .generation = object.generation};
}

std::optional<std::int32_t> StateDirectory::take_device_slot() noexcept {
    if (free_device_count_ == 0) { return std::nullopt; }
    return free_device_slots_[--free_device_count_];
}

void StateDirectory::return_device_slot(std::int32_t slot) noexcept {
    free_device_slots_[free_device_count_++] = slot;
}

std::uint64_t StateDirectory::next_epoch() noexcept {
    if (++next_content_epoch_ == 0) { ++next_content_epoch_; }
    return next_content_epoch_;
}

std::uint64_t StateDirectory::next_transfer() noexcept {
    if (++next_transfer_id_ == 0) { ++next_transfer_id_; }
    return next_transfer_id_;
}

StateDirectory::Object& StateDirectory::require(StateId handle) {
    if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
    return objects_[handle.index];
}

const StateDirectory::Object& StateDirectory::require(StateId handle) const {
    if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
    return objects_[handle.index];
}

bool StateDirectory::valid_transfer(const Transfer& transfer) const noexcept {
    if (transfer.owner_ != this || transfer.id_ == 0 || !valid(transfer.source_.image()) ||
        !valid(transfer.destination_)) {
        return false;
    }
    const Object& destination = objects_[transfer.destination_.index];
    return destination.transfer_id == transfer.id_;
}

void StateDirectory::validate_transfer(const Transfer& transfer) const {
    if (!valid_transfer(transfer)) {
        throw std::logic_error("StateImage transfer capability is stale");
    }
}

void StateDirectory::release_host_slot(std::optional<HostAllocation>& slot) noexcept {
    if (!slot) return;
    slot.reset();
    --host_images_;
}

void StateDirectory::consume(Transfer& transfer) noexcept {
    transfer.owner_ = nullptr;
    transfer.id_    = 0;
}

} // namespace ninfer::models::context
