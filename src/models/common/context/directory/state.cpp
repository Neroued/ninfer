#include "models/common/context/directory/state.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {

StateDirectory::ReadLease::~ReadLease() { reset(); }

StateDirectory::ReadLease::ReadLease(ReadLease&& other) noexcept
    : directory_(std::exchange(other.directory_, nullptr)), image_(other.image_),
      epoch_(other.epoch_), memory_(other.memory_), use_(other.use_) {}

StateDirectory::ReadLease& StateDirectory::ReadLease::operator=(ReadLease&& other) noexcept {
    if (this != &other) {
        reset();
        directory_ = std::exchange(other.directory_, nullptr);
        image_     = other.image_;
        epoch_     = other.epoch_;
        memory_    = other.memory_;
        use_       = other.use_;
    }
    return *this;
}

void StateDirectory::ReadLease::reset() noexcept {
    if (!directory_) return;
    auto* store = std::exchange(directory_, nullptr);
    auto& image = store->require(image_);
    if (!image.source_pins || image.content_epoch != epoch_) std::terminate();
    --image.source_pins;
    if (use_ == ContextUse::Request) store->release_request_reference(image_, memory_);
    store->retire_released_owner(image_);
}

std::int64_t StateDirectory::ReadLease::device_offset(std::uint32_t rank) const {
    if (!directory_ || directory_->content_epoch(image_) != epoch_)
        throw std::logic_error("StateImage reader has no current content");
    return directory_->physical_offset(image_, rank);
}

ContextRecordBinding StateDirectory::ReadLease::host_binding(std::uint32_t rank) const {
    if (!directory_ || directory_->content_epoch(image_) != epoch_)
        throw std::logic_error("StateImage reader has no current content");
    return directory_->host_binding(image_, rank);
}

StateDirectory::ReadLease StateDirectory::read(StateId handle, ContextMemory memory,
                                               ContextUse use) {
    auto& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable ||
        object.source_pins == std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("StateImage is not readable as an immutable replica");
    ++object.source_pins;
    if (use == ContextUse::Request) retain_request_reference(handle, memory);
    return ReadLease(*this, handle, object.content_epoch, memory, use);
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

StateSelectors StateDirectory::Destination::begin_write(StateId source, std::uint32_t rank) {
    const auto selectors = directory_->begin_write(source, image_, rank);
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
      destination_(other.destination_), records_(std::move(other.records_)) {}

StateTransferDescription StateDirectory::Transfer::description(std::uint32_t rank) const {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->validate_transfer(*this);
    const auto target = std::find_if(records_.begin(), records_.end(), [rank](const auto& record) {
        return record.domain().rank == rank;
    });
    if (target == records_.end())
        throw std::invalid_argument("StateImage transfer has no rank part");
    return {
        .direction          = direction_,
        .source             = source_.image(),
        .destination        = destination_,
        .source_record      = direction_ == StateCopyDirection::DeviceToHost
                                  ? owner_->device_binding(source_.image(), rank)
                                  : owner_->host_binding(source_.image(), rank),
        .destination_record = target->binding(),
        .epoch              = source_.epoch(),
    };
}

std::vector<std::uint32_t> StateDirectory::Transfer::ranks() const {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->validate_transfer(*this);
    std::vector<std::uint32_t> result;
    result.reserve(records_.size());
    for (const auto& record : records_) result.push_back(record.domain().rank);
    return result;
}

TransferWork StateDirectory::Transfer::work() const {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->validate_transfer(*this);
    TransferWork result{};
    for (const auto& record : records_) {
        result.payload_bytes += record.binding().bytes;
        ++result.copy_operations;
    }
    return result;
}

void StateDirectory::Transfer::publish(bool keep_source_replica) {
    if (!owner_) throw std::logic_error("StateImage copy already consumed");
    owner_->publish_transfer(*this, keep_source_replica);
}

std::uint32_t state_record_layout(std::uint32_t variant) {
    if (variant >= 0x80000000U)
        throw std::invalid_argument("State variant exceeds layout ID range");
    return variant == 0 ? 0 : 0x80000000U | variant;
}

StateDirectory::StateDirectory(StateDirectorySpec spec, RecordDomains& records,
                               MetadataBudget& metadata)
    : spec_(std::move(spec)), records_(&records), metadata_account_(metadata),
      metadata_budget_(&metadata), objects_(spec_.logical_capacity, metadata) {
    if (spec_.logical_capacity == 0 || spec_.layouts.empty()) {
        throw std::invalid_argument("StateDirectory requires capacity and layouts");
    }
    for (std::size_t index = 0; index < spec_.layouts.size(); ++index) {
        const auto& variant       = spec_.layouts[index];
        const auto& record_layout = records.layout(state_record_layout(variant.variant));
        if (variant.parts.empty() || variant.parts.size() != record_layout.parts.size() ||
            std::any_of(spec_.layouts.begin(), spec_.layouts.begin() + index,
                        [&](const auto& other) { return variant.variant == other.variant; })) {
            throw std::invalid_argument("StateDirectory layout inventory is inconsistent");
        }
        max_parts_ = std::max(max_parts_, variant.parts.size());
        for (std::size_t i = 0; i < variant.parts.size(); ++i) {
            const auto& declared = variant.parts[i];
            const auto found =
                std::find_if(record_layout.parts.begin(), record_layout.parts.end(),
                             [&](const auto& item) { return item.rank == declared.rank; });
            if (std::any_of(variant.parts.begin(), variant.parts.begin() + i,
                            [&](const auto& item) { return item.rank == declared.rank; }) ||
                found == record_layout.parts.end() || declared.image_bytes == 0 ||
                found->record.bytes != declared.image_bytes ||
                found->record.alignment != declared.alignment) {
                throw std::invalid_argument("StateDirectory record layout is inconsistent");
            }
        }
    }
    auto bytes = sizeof(*this) + spec_.layouts.capacity() * sizeof(StateLayoutSpec);
    for (const auto& layout : spec_.layouts) {
        bytes += layout.parts.capacity() * sizeof(StatePartSpec);
    }
    metadata_account_.acquire(bytes);
}

const StateLayoutSpec& StateDirectory::layout_spec(std::uint32_t variant) const {
    const auto found =
        std::find_if(spec_.layouts.begin(), spec_.layouts.end(),
                     [variant](const auto& item) { return item.variant == variant; });
    if (found == spec_.layouts.end()) throw std::invalid_argument("Unknown State layout variant");
    return *found;
}

std::uint32_t StateDirectory::variant(StateId image) const { return require(image).variant; }

std::size_t StateDirectory::object_metadata_bytes(std::size_t parts) {
    if (parts > (SIZE_MAX - sizeof(Object) - sizeof(std::uint32_t)) / sizeof(Object::Part)) {
        throw std::overflow_error("StateDirectory metadata overflows size_t");
    }
    return sizeof(Object) + sizeof(std::uint32_t) + parts * sizeof(Object::Part);
}

std::size_t StateDirectory::metadata_bytes(const StateDirectorySpec& spec) {
    std::size_t parts = 0;
    auto fixed        = sizeof(StateDirectory) + spec.layouts.size() * sizeof(StateLayoutSpec);
    for (const auto& layout : spec.layouts) {
        parts = std::max(parts, layout.parts.size());
        fixed += layout.parts.size() * sizeof(StatePartSpec);
    }
    const auto object   = object_metadata_bytes(parts);
    const auto segments = (static_cast<std::size_t>(spec.logical_capacity) + 255) / 256;
    fixed += segments * sizeof(std::unique_ptr<Object[]>);
    if (spec.logical_capacity > (SIZE_MAX - fixed) / object) {
        throw std::overflow_error("StateDirectory metadata overflows size_t");
    }
    return fixed + static_cast<std::size_t>(spec.logical_capacity) * object;
}

std::size_t StateDirectory::allocated_metadata_bytes() const {
    return metadata_account_.bytes() + objects_.storage_bytes();
}

std::uint32_t StateDirectory::available_descriptors() const noexcept {
    return capacity() - occupied();
}

std::size_t StateDirectory::metadata_requirement(std::uint32_t count) const {
    if (count > available_descriptors()) return SIZE_MAX;
    std::uint32_t highwater = initialized_objects_;
    std::size_t bytes = 0, reallocation = 0;
    if (count > free_object_count_) {
        const auto required = occupied() + count;
        highwater           = static_cast<std::uint32_t>(std::max<std::uint64_t>(
            objects_.size(),
            std::min<std::uint64_t>(capacity(), (std::uint64_t(required) + 255) / 256 * 256)));
        bytes += objects_.growth_bytes(required);
        if (highwater > free_objects_.capacity()) {
            bytes += (highwater - free_objects_.capacity()) * sizeof(std::uint32_t);
            reallocation = free_objects_.capacity() * sizeof(std::uint32_t);
        }
    }
    const auto added = highwater - initialized_objects_;
    std::vector<std::uint32_t> holes;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index =
            i < added ? highwater - 1 - i : free_objects_[free_object_count_ - 1 - (i - added)];
        const bool present = objects_.is_allocated(index);
        if (!present && index < objects_.size()) {
            const auto segment = index / 256;
            if (std::find(holes.begin(), holes.end(), segment) == holes.end()) {
                holes.push_back(segment);
                bytes += objects_.ensure_bytes(index);
            }
        }
        const auto parts = present ? objects_[index].parts.capacity() : 0;
        if (parts < max_parts_) {
            bytes += (max_parts_ - parts) * sizeof(Object::Part);
            reallocation = std::max(reallocation, parts * sizeof(Object::Part));
        }
    }
    return bytes + reallocation;
}

bool StateDirectory::reserve_metadata(std::uint32_t count) {
    if (count > available_descriptors()) return false;
    const auto required = metadata_requirement(count);
    if (required > metadata_budget_->available()) throw MetadataExhausted(required);
    if (count > free_object_count_) {
        objects_.grow(occupied() + count);
        metadata_account_.reserve(free_objects_, objects_.size());
        free_objects_.resize(objects_.size());
        while (initialized_objects_ < objects_.size()) {
            free_objects_[free_object_count_++] = initialized_objects_++;
        }
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index = free_objects_[free_object_count_ - 1 - i];
        objects_.ensure(index);
        auto& object = objects_[index];
        metadata_account_.reserve(object.parts, max_parts_);
    }
    return true;
}

MetadataReservation StateDirectory::reserve_descriptors(std::uint32_t count) {
    if (!reserve_metadata(count))
        throw std::runtime_error("State metadata descriptor limit exhausted");
    std::vector<std::uint32_t> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index  = free_objects_[--free_object_count_];
        auto& object      = objects_[index];
        object.role       = StateRole::ReservedDestination;
        object.generation = 0;
        ids.push_back(index);
    }
    return MetadataReservation(*this, std::move(ids));
}

void StateDirectory::release_metadata(std::span<const std::uint32_t> ids) noexcept {
    for (const auto index : ids) {
        if (!objects_.is_allocated(index)) std::terminate();
        auto& object = objects_[index];
        if (object.role != StateRole::ReservedDestination || object.generation != 0)
            std::terminate();
        object.role                         = StateRole::Free;
        free_objects_[free_object_count_++] = index;
    }
}

std::size_t StateDirectory::reclaimable_metadata(std::span<const StateId> released) const {
    std::size_t bytes = 0;
    for (std::uint32_t first = 0; first < objects_.size(); first += 256) {
        if (!objects_.is_allocated(first)) continue;
        const auto end      = first + std::min(256U, objects_.size() - first);
        std::size_t segment = 0;
        for (auto i = first; i < end; ++i) {
            const auto& object = objects_[i];
            if (object.role != StateRole::Free &&
                std::find(released.begin(), released.end(), StateId{i, object.generation}) ==
                    released.end()) {
                segment = 0;
                break;
            }
            segment += sizeof(Object) + object.parts.capacity() * sizeof(Object::Part);
        }
        bytes += segment;
    }
    return bytes;
}

std::size_t StateDirectory::reclaim_metadata() {
    const auto before = allocated_metadata_bytes();
    for (std::uint32_t first = 0; first < objects_.size(); first += 256) {
        if (!objects_.is_allocated(first)) continue;
        const auto end = first + std::min<std::uint32_t>(256, objects_.size() - first);
        bool empty     = true;
        for (auto index = first; index < end; ++index) {
            if (objects_[index].role != StateRole::Free) {
                empty = false;
                break;
            }
        }
        if (!empty) continue;
        std::size_t parts = 0;
        for (auto index = first; index < end; ++index) {
            parts += objects_[index].parts.capacity() * sizeof(Object::Part);
        }
        metadata_account_.release(parts);
        objects_.release_segment(first);
    }
    return before - allocated_metadata_bytes();
}

std::size_t StateDirectory::image_bytes(std::uint32_t rank, std::uint32_t variant) const {
    const auto& parts = layout_spec(variant).parts;
    const auto found  = std::find_if(parts.begin(), parts.end(),
                                     [rank](const auto& value) { return value.rank == rank; });
    if (found == parts.end()) throw std::invalid_argument("StateImage has no rank part");
    return found->image_bytes;
}

std::vector<std::uint32_t> StateDirectory::ranks(std::uint32_t variant) const {
    std::vector<std::uint32_t> result;
    const auto& parts = layout_spec(variant).parts;
    result.reserve(parts.size());
    for (const auto& part : parts) result.push_back(part.rank);
    return result;
}

std::uint32_t StateDirectory::capacity() const noexcept { return objects_.limit(); }

std::uint32_t StateDirectory::occupied() const noexcept {
    return initialized_objects_ - free_object_count_;
}

std::uint32_t StateDirectory::device_occupied() const noexcept { return device_images_; }

std::uint32_t StateDirectory::host_occupied() const noexcept { return host_images_; }

std::optional<StateId> StateDirectory::reserve_destination(RecordReservation reservation,
                                                           std::uint32_t variant) {
    const auto& parts = layout_spec(variant).parts;
    if (!reservation.ready()) return std::nullopt;
    if (next_generation_ == UINT32_MAX)
        throw std::overflow_error("State identity generation exhausted");
    if (reservation.ranks().size() != parts.size()) {
        throw std::invalid_argument("StateImage destination reservation has incompatible parts");
    }
    for (const auto& part : parts) {
        if (reservation.remaining(part.rank) != 1) {
            throw std::invalid_argument(
                "StateImage destination reservation must contain one image");
        }
    }
    auto records = reservation.materialize();
    validate_records(records, ContextMemory::Device, variant);
    for (auto& record : records) records_->classify(record, state_record_layout(variant));
    const auto index = reservation.take_metadata(*this);
    return allocate(index, StateRole::ReservedDestination, variant, std::move(records));
}

std::optional<StateId> StateDirectory::reserve_destination(std::uint32_t variant) {
    (void)layout_spec(variant);
    if (occupied() == capacity()) return std::nullopt;
    auto metadata = reserve_descriptors(1);
    auto reserved =
        records_->reserve(RecordDemand{state_record_layout(variant), ContextMemory::Device, 1, {}});
    if (!reserved) return std::nullopt;
    reserved->attach_metadata(std::move(metadata));
    return reserve_destination(std::move(*reserved), variant);
}

std::optional<StateId> StateDirectory::reserve_logical_destination(std::uint32_t variant) {
    (void)layout_spec(variant);
    if (occupied() == capacity()) return std::nullopt;
    if (next_generation_ == UINT32_MAX)
        throw std::overflow_error("State identity generation exhausted");
    auto metadata = reserve_descriptors(1);
    return allocate(metadata.take(*this), StateRole::ReservedDestination, variant);
}

std::optional<StateId> StateDirectory::reserve_logical_destination(RecordReservation& reservation,
                                                                   std::uint32_t variant) {
    (void)layout_spec(variant);
    if (next_generation_ == UINT32_MAX)
        throw std::overflow_error("State identity generation exhausted");
    return allocate(reservation.take_metadata(*this), StateRole::ReservedDestination, variant);
}

void StateDirectory::complete_initialization(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::ReservedDestination || !device_resident(handle) ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage initialization has no exclusive reservation");
    }
    if (object.content_epoch == 0) { object.content_epoch = next_epoch(); }
    object.role = StateRole::ActiveMutable;
    refresh_usage(object);
}

void StateDirectory::complete_conversion(const ReadLease& source, StateId destination) {
    if (source.directory_ != this || !source.image_.valid())
        throw std::invalid_argument("State conversion source belongs to another directory");
    const auto& from = require(source.image_);
    auto& to         = require(destination);
    if (source.image_ == destination || from.role != StateRole::CheckpointImmutable ||
        from.content_epoch != source.epoch_ || !source.epoch_ || !from.source_pins ||
        to.role != StateRole::ReservedDestination || !device_resident(destination) ||
        to.content_epoch || to.source_pins || to.destination_pinned || has_pending_replica(to))
        throw std::logic_error("State conversion has no exclusive destination or current source");
    to.content_epoch = source.epoch_;
    to.role          = StateRole::ActiveMutable;
}

bool StateDirectory::valid(StateId handle) const noexcept {
    return handle.valid() && handle.index < initialized_objects_ &&
           objects_.is_allocated(handle.index) && objects_[handle.index].role != StateRole::Free &&
           objects_[handle.index].generation == handle.generation;
}

std::uint32_t StateDirectory::descriptor_index(StateId handle) const {
    (void)require(handle);
    return handle.index;
}

StateRole StateDirectory::role(StateId handle) const { return require(handle).role; }

StateResidency StateDirectory::residency(StateId handle) const {
    const bool device = device_resident(handle);
    const bool host   = host_resident(handle);
    if (device && host) return StateResidency::Both;
    if (device) return StateResidency::DeviceOnly;
    if (host) return StateResidency::HostOnly;
    return StateResidency::None;
}

bool StateDirectory::device_resident(StateId handle) const {
    const auto& object = require(handle);
    return std::all_of(object.parts.begin(), object.parts.end(),
                       [](const auto& part) { return part.device.has_value(); });
}

bool StateDirectory::host_resident(StateId handle) const {
    const auto& object = require(handle);
    return std::all_of(object.parts.begin(), object.parts.end(),
                       [](const auto& part) { return part.host.has_value(); });
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

void StateDirectory::refresh_usage(Object& object) noexcept {
    const bool active = object.role == StateRole::ActiveMutable ||
                        object.role == StateRole::ReservedDestination ||
                        object.role == StateRole::Writing;
    for (auto& local : object.parts) {
        if (local.device)
            local.device->set_use(active || object.request_references[0] ? ContextUse::Request
                                                                         : ContextUse::Cache);
        if (local.host)
            local.host->set_use(object.request_references[1] ? ContextUse::Request
                                                             : ContextUse::Cache);
    }
}

void StateDirectory::retain_request_reference(StateId handle, ContextMemory memory) {
    auto& object = require(handle);
    auto& count  = object.request_references[static_cast<std::size_t>(memory)];
    if (count == UINT32_MAX) throw std::overflow_error("StateImage request reference overflow");
    if (++count == 1) refresh_usage(object);
}

void StateDirectory::release_request_reference(StateId handle, ContextMemory memory) noexcept {
    auto& object = require(handle);
    auto& count  = object.request_references[static_cast<std::size_t>(memory)];
    if (!count) std::terminate();
    if (--count == 0) refresh_usage(object);
}

void StateDirectory::retain_checkpoint_reference(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable ||
        (object.release_pending && !object.source_pins) ||
        object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("StateImage checkpoint reference is not retainable");
    }
    if (object.release_pending) {
        for (auto& local : object.parts) {
            if (local.device) local.device->cancel_retirement();
            if (local.host) local.host->cancel_retirement();
        }
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
        for (auto& local : object.parts) {
            if (local.device) local.device->defer_retirement();
            if (local.host) local.host->defer_retirement();
        }
        retire_released_owner(handle);
    }
    return true;
}

StateDirectory::Object::Part& StateDirectory::part(Object& object, std::uint32_t rank) {
    return const_cast<Object::Part&>(part(std::as_const(object), rank));
}

const StateDirectory::Object::Part& StateDirectory::part(const Object& object, std::uint32_t rank) {
    const auto found = std::find_if(object.parts.begin(), object.parts.end(),
                                    [rank](const auto& value) { return value.rank == rank; });
    if (found == object.parts.end()) throw std::invalid_argument("StateImage has no rank part");
    return *found;
}

std::int64_t StateDirectory::physical_offset(StateId handle, std::uint32_t rank) const {
    return device_binding(handle, rank).offset;
}

ContextRecordBinding StateDirectory::device_binding(StateId handle, std::uint32_t rank) const {
    const auto& value = part(require(handle), rank);
    if (!value.device) throw std::logic_error("StateImage rank part has no Device storage");
    return value.device->binding();
}

ContextRecordBinding StateDirectory::host_binding(StateId handle, std::uint32_t rank) const {
    const auto& value = part(require(handle), rank);
    if (!value.host) throw std::logic_error("StateImage rank part has no Host storage");
    return value.host->binding();
}

std::vector<RecordRelease> StateDirectory::allocations(StateId handle) const {
    const auto& object = require(handle);
    std::vector<RecordRelease> result;
    result.reserve(object.parts.size() * 2);
    for (const auto& part : object.parts) {
        for (const auto* record : {&part.device, &part.host}) {
            if (*record) result.push_back({(*record)->domain(), (*record)->allocation()});
        }
    }
    return result;
}

void StateDirectory::make_mutable(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable || !device_resident(handle) ||
        object.checkpoint_references != 0 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint is not movable");
    }
    if (host_resident(handle)) --host_images_;
    for (auto& part : object.parts) part.host.reset();
    object.role = StateRole::ActiveMutable;
    refresh_usage(object);
}

void StateDirectory::consume_checkpoint(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::CheckpointImmutable || !device_resident(handle) ||
        object.checkpoint_references != 1 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint has other owners or readers");
    }
    object.checkpoint_references = 0;
    object.release_pending       = false;
    if (host_resident(handle)) --host_images_;
    for (auto& part : object.parts) part.host.reset();
    object.role = StateRole::ActiveMutable;
    refresh_usage(object);
}

bool StateDirectory::drop_device_replica(StateId handle) noexcept {
    if (!valid(handle)) return false;
    auto& object = objects_[handle.index];
    if (object.role != StateRole::CheckpointImmutable || !device_resident(handle) ||
        !host_resident(handle) || object.source_pins || object.destination_pinned ||
        has_pending_replica(object))
        return false;
    for (auto& part : object.parts) part.device.reset();
    --device_images_;
    return true;
}

bool StateDirectory::drop_host_replica(StateId handle) noexcept {
    if (!valid(handle)) return false;
    auto& object = objects_[handle.index];
    if (object.role != StateRole::CheckpointImmutable || !device_resident(handle) ||
        !host_resident(handle) || object.source_pins || object.destination_pinned ||
        has_pending_replica(object))
        return false;
    for (auto& part : object.parts) part.host.reset();
    --host_images_;
    return true;
}

void StateDirectory::split_device_replica_identity(StateId source, StateId destination) {
    auto& from = require(source);
    auto& to   = require(destination);
    if (source == destination || from.role != StateRole::CheckpointImmutable ||
        from.variant != to.variant || !device_resident(source) || !host_resident(source) ||
        from.source_pins || from.destination_pinned || has_pending_replica(from) ||
        to.role != StateRole::ReservedDestination ||
        residency(destination) != StateResidency::None || has_pending_replica(to) ||
        to.source_pins || to.destination_pinned) {
        throw std::logic_error("StateImage replica identity split is invalid");
    }
    for (auto& source_part : from.parts) {
        part(to, source_part.rank).device = std::move(source_part.device);
        source_part.device.reset();
    }
    to.content_epoch = from.content_epoch;
    refresh_usage(to);
}

void StateDirectory::freeze(StateId handle) {
    Object& object = require(handle);
    if (object.role != StateRole::ActiveMutable || !device_resident(handle) ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage active image is not freezable");
    }
    object.role = StateRole::CheckpointImmutable;
    refresh_usage(object);
}

StateSelectors StateDirectory::begin_write(StateId source, StateId destination,
                                           std::uint32_t rank) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source == destination || source_object.variant != destination_object.variant ||
        source_object.role != StateRole::CheckpointImmutable || !device_resident(source) ||
        source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
        destination_object.role != StateRole::ReservedDestination ||
        !device_resident(destination) || destination_object.source_pins != 0 ||
        destination_object.destination_pinned || has_pending_replica(destination_object)) {
        throw std::logic_error("StateImage fork binding is invalid");
    }
    const StateSelectors result{physical_offset(source, rank), physical_offset(destination, rank)};
    ++source_object.source_pins;
    retain_request_reference(source, ContextMemory::Device);
    destination_object.destination_pinned = true;
    destination_object.role               = StateRole::Writing;
    destination_object.write_source       = source;
    destination_object.content_epoch      = source_object.content_epoch;
    return result;
}

void StateDirectory::complete_write(StateId source, StateId destination) {
    if (source == destination) {
        (void)selectors(source, destination, layout_spec(variant(source)).parts.front().rank);
        require(destination).content_epoch = next_epoch();
        return;
    }
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source_object.role != StateRole::CheckpointImmutable || source_object.source_pins == 0 ||
        destination_object.role != StateRole::Writing || !device_resident(destination) ||
        !destination_object.destination_pinned || destination_object.write_source != source) {
        throw std::logic_error("StateImage fork commit is invalid");
    }
    --source_object.source_pins;
    release_request_reference(source, ContextMemory::Device);
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
    release_request_reference(source, ContextMemory::Device);
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
           device_resident(destination) && destination_object.destination_pinned &&
           destination_object.write_source == source;
}

StateSelectors StateDirectory::selectors(StateId source, StateId destination,
                                         std::uint32_t rank) const {
    const Object& source_object      = require(source);
    const Object& destination_object = require(destination);
    const bool inplace               = source == destination;
    if (!device_resident(source) || !device_resident(destination) ||
        (inplace && (source_object.role != StateRole::ActiveMutable ||
                     source_object.source_pins != 0 || source_object.destination_pinned)) ||
        (!inplace &&
         (source_object.role != StateRole::CheckpointImmutable || source_object.source_pins == 0 ||
          destination_object.role != StateRole::Writing || !destination_object.destination_pinned ||
          destination_object.write_source != source))) {
        throw std::logic_error("StateImage execution binding is invalid");
    }
    return {.source      = physical_offset(source, rank),
            .destination = physical_offset(destination, rank)};
}

std::optional<StateDirectory::Transfer>
StateDirectory::reserve_device_to_host(StateId source, RecordReservation reserved) {
    auto& object = require(source);
    if (object.role != StateRole::CheckpointImmutable || !device_resident(source) ||
        host_resident(source) || has_pending_replica(object) || object.source_pins == UINT32_MAX)
        return std::nullopt;
    if (reserved.empty()) {
        auto target = records_->reserve(
            RecordDemand{state_record_layout(object.variant), ContextMemory::Host, 1, {}});
        if (!target) return std::nullopt;
        reserved = std::move(*target);
    }
    if (!reserved.ready()) return std::nullopt;
    auto targets = reserved.materialize();
    validate_records(targets, ContextMemory::Host, object.variant);
    for (auto& record : targets) records_->classify(record, state_record_layout(object.variant));
    const auto id  = next_transfer();
    const auto use = object.request_references[1] ? ContextUse::Request : ContextUse::Cache;
    for (auto& record : targets) record.set_use(use);
    Transfer transfer(*this, id, StateCopyDirection::DeviceToHost,
                      read(source, ContextMemory::Device, use), source);
    transfer.records_ = std::move(targets);
    ++host_images_;
    object.transfer_id        = id;
    object.destination_pinned = true;
    return transfer;
}

std::optional<StateDirectory::Transfer> StateDirectory::reserve_host_to_device(StateId source) {
    return reserve_host_to_device(source, source);
}

std::optional<StateDirectory::Transfer>
StateDirectory::reserve_host_to_device(StateId source, StateId destination,
                                       RecordReservation reserved) {
    auto& from = require(source);
    auto& to   = require(destination);
    if (from.variant != to.variant || !host_resident(source) ||
        from.role != StateRole::CheckpointImmutable || from.source_pins == UINT32_MAX ||
        device_resident(destination) || has_pending_replica(to) || to.destination_pinned ||
        (source != destination && (to.role != StateRole::ReservedDestination || to.source_pins ||
                                   host_resident(destination)))) {
        return std::nullopt;
    }
    if (reserved.empty()) {
        auto target = records_->reserve(
            RecordDemand{state_record_layout(from.variant), ContextMemory::Device, 1, {}});
        if (!target) return std::nullopt;
        reserved = std::move(*target);
    }
    if (!reserved.ready()) return std::nullopt;
    auto targets = reserved.materialize();
    validate_records(targets, ContextMemory::Device, from.variant);
    for (auto& record : targets) records_->classify(record, state_record_layout(from.variant));
    const auto id = next_transfer();
    Transfer transfer(*this, id, StateCopyDirection::HostToDevice,
                      read(source, ContextMemory::Host), destination);
    transfer.records_ = std::move(targets);
    ++device_images_;
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
    if (source.content_epoch != transfer.source_.epoch()) {
        throw std::logic_error("StateImage source changed during migration");
    }
    if (!keep_source_replica && source_handle == transfer.destination_ && source.source_pins != 1) {
        throw std::logic_error("StateImage source replica still has readers");
    }
    for (auto& record : transfer.records_) {
        auto& local = part(destination, record.domain().rank);
        if (transfer.direction_ == StateCopyDirection::DeviceToHost) {
            local.host.emplace(std::move(record));
            if (!keep_source_replica) local.device.reset();
        } else {
            local.device.emplace(std::move(record));
            if (!keep_source_replica && source_handle == transfer.destination_) local.host.reset();
        }
    }
    if (transfer.direction_ == StateCopyDirection::DeviceToHost && !keep_source_replica)
        --device_images_;
    if (transfer.direction_ == StateCopyDirection::HostToDevice && !keep_source_replica &&
        source_handle == transfer.destination_)
        --host_images_;
    destination.destination_pinned = false;
    destination.transfer_id        = 0;
    refresh_usage(destination);
    consume(transfer);
    transfer.source_.reset();
}

void StateDirectory::abort_transfer(Transfer& transfer) noexcept {
    if (!valid_transfer(transfer)) std::terminate();
    auto& destination = objects_[transfer.destination_.index];
    transfer.records_.clear();
    if (transfer.direction_ == StateCopyDirection::DeviceToHost)
        --host_images_;
    else
        --device_images_;
    destination.destination_pinned = false;
    destination.transfer_id        = 0;
    if (transfer.source_.image() != transfer.destination_) destination.content_epoch = 0;
    refresh_usage(destination);
    consume(transfer);
    transfer.source_.reset();
}

bool StateDirectory::release(StateId handle) noexcept {
    if (!can_release(handle)) return false;
    auto& object = objects_[handle.index];
    if (host_resident(handle)) --host_images_;
    if (device_resident(handle)) --device_images_;
    for (auto& local : object.parts) {
        local.host.reset();
        local.device.reset();
    }
    object.role            = StateRole::Free;
    object.content_epoch   = 0;
    object.release_pending = false;

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

void StateDirectory::validate_records(const std::vector<RecordLease>& records, ContextMemory memory,
                                      std::uint32_t variant) const {
    const auto& parts = layout_spec(variant).parts;
    if (records.size() != parts.size()) {
        throw std::invalid_argument("StateImage reservation does not contain all rank parts");
    }
    for (const auto& required : parts) {
        const RecordLease* found = nullptr;
        for (const auto& record : records) {
            if (record.domain().rank != required.rank) continue;
            if (found) throw std::invalid_argument("StateImage reservation repeats a rank part");
            found = &record;
        }
        if (!found || !found->valid())
            throw std::invalid_argument("StateImage rank part is missing");
        const auto binding = found->binding();
        if (binding.domain.memory != memory || binding.bytes != required.image_bytes ||
            binding.offset % required.alignment != 0 ||
            !found->belongs_to(records_->pool(binding.domain))) {
            throw std::invalid_argument("StateImage reservation has incompatible storage");
        }
    }
}

std::optional<StateId> StateDirectory::allocate(std::uint32_t index, StateRole role,
                                                std::uint32_t variant,
                                                std::vector<RecordLease> records) {
    const auto& parts = layout_spec(variant).parts;
    auto& object      = objects_[index];
    object.parts.clear();
    for (const auto& spec : parts) object.parts.push_back({spec.rank, {}, {}});
    object.role       = role;
    object.variant    = variant;
    object.generation = ++next_generation_;
    if (!records.empty()) ++device_images_;
    for (auto& record : records)
        part(object, record.domain().rank).device.emplace(std::move(record));
    return StateId{index, object.generation};
}

std::optional<StateId> StateDirectory::image_for_allocation(AllocationId allocation) const {
    for (std::uint32_t index = 0; index < initialized_objects_; ++index) {
        if (!objects_.is_allocated(index)) continue;
        const auto& object = objects_[index];
        if (object.role == StateRole::Free) continue;
        for (const auto& part : object.parts) {
            for (const auto* record : {&part.device, &part.host}) {
                if (*record && (*record)->allocation() == allocation) {
                    return StateId{index, object.generation};
                }
            }
        }
    }
    return std::nullopt;
}

StateDirectory::ReadLease StateDirectory::pin_binding(StateId image) {
    auto& object = require(image);
    if (object.source_pins == UINT32_MAX) {
        throw std::overflow_error("State binding pin count exhausted");
    }
    ++object.source_pins;
    return ReadLease(*this, image, object.content_epoch, ContextMemory::Device, ContextUse::Cache);
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

void StateDirectory::consume(Transfer& transfer) noexcept {
    transfer.owner_ = nullptr;
    transfer.id_    = 0;
}

} // namespace ninfer::models::context
